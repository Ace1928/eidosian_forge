import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True), '\n        min_length_for_response (`int`, *optional*, defaults to 32):\n            The minimum length (in number of tokens) for a response.\n        minimum_tokens (`int`, *optional*, defaults to 10):\n            The minimum length of tokens to leave for a response.')
class ConversationalPipeline(Pipeline):
    """
    Multi-turn conversational pipeline.

    Example:

    ```python
    >>> from transformers import pipeline, Conversation
    # Any model with a chat template can be used in a ConversationalPipeline.

    >>> chatbot = pipeline(model="facebook/blenderbot-400M-distill")
    >>> # Conversation objects initialized with a string will treat it as a user message
    >>> conversation = Conversation("I'm looking for a movie - what's your favourite one?")
    >>> conversation = chatbot(conversation)
    >>> conversation.messages[-1]["content"]
    "I don't really have a favorite movie, but I do like action movies. What about you?"

    >>> conversation.add_message({"role": "user", "content": "That's interesting, why do you like action movies?"})
    >>> conversation = chatbot(conversation)
    >>> conversation.messages[-1]["content"]
    " I think it's just because they're so fast-paced and action-fantastic."
    ```

    Learn more about the basics of using a pipeline in the [pipeline tutorial](../pipeline_tutorial)

    This conversational pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"conversational"`.

    This pipeline can be used with any model that has a [chat
    template](https://huggingface.co/docs/transformers/chat_templating) set.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn('`ConversationalPipeline` is now deprecated, and the functionality has been moved to the standard `text-generation` pipeline, which now accepts lists of message dicts as well as strings. This class will be removed in v4.42.', DeprecationWarning)
        super().__init__(*args, **kwargs)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _sanitize_parameters(self, min_length_for_response=None, minimum_tokens=None, clean_up_tokenization_spaces=None, **generate_kwargs):
        preprocess_params = {}
        forward_params = {}
        postprocess_params = {}
        if min_length_for_response is not None:
            preprocess_params['min_length_for_response'] = min_length_for_response
        if minimum_tokens is not None:
            forward_params['minimum_tokens'] = minimum_tokens
        if 'max_length' in generate_kwargs:
            forward_params['max_length'] = generate_kwargs['max_length']
        if clean_up_tokenization_spaces is not None:
            postprocess_params['clean_up_tokenization_spaces'] = clean_up_tokenization_spaces
        if generate_kwargs:
            forward_params.update(generate_kwargs)
        return (preprocess_params, forward_params, postprocess_params)

    def __call__(self, conversations: Union[List[Dict], Conversation, List[Conversation]], num_workers=0, **kwargs):
        """
        Generate responses for the conversation(s) given as inputs.

        Args:
            conversations (a [`Conversation`] or a list of [`Conversation`]):
                Conversation to generate responses for. Inputs can also be passed as a list of dictionaries with `role`
                and `content` keys - in this case, they will be converted to `Conversation` objects automatically.
                Multiple conversations in either format may be passed as a list.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the potential extra spaces in the text output.
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Returns:
            [`Conversation`] or a list of [`Conversation`]: Conversation(s) with updated generated responses for those
            containing a new user input.
        """
        if isinstance(conversations, list) and isinstance(conversations[0], dict):
            conversations = Conversation(conversations)
        elif isinstance(conversations, list) and isinstance(conversations[0], list):
            conversations = [Conversation(conv) for conv in conversations]
        outputs = super().__call__(conversations, num_workers=num_workers, **kwargs)
        if isinstance(outputs, list) and len(outputs) == 1:
            return outputs[0]
        return outputs

    def preprocess(self, conversation: Conversation, min_length_for_response=32) -> Dict[str, Any]:
        input_ids = self.tokenizer.apply_chat_template(conversation, add_generation_prompt=True)
        if self.framework == 'pt':
            input_ids = torch.LongTensor([input_ids])
        elif self.framework == 'tf':
            input_ids = tf.constant([input_ids])
        return {'input_ids': input_ids, 'conversation': conversation}

    def _forward(self, model_inputs, minimum_tokens=10, **generate_kwargs):
        n = model_inputs['input_ids'].shape[1]
        conversation = model_inputs.pop('conversation')
        if 'max_length' not in generate_kwargs and 'max_new_tokens' not in generate_kwargs:
            generate_kwargs['max_new_tokens'] = 256
        output_ids = self.model.generate(**model_inputs, **generate_kwargs)
        if self.model.config.is_encoder_decoder:
            start_position = 1
        else:
            start_position = n
        return {'output_ids': output_ids[:, start_position:], 'conversation': conversation}

    def postprocess(self, model_outputs, clean_up_tokenization_spaces=True):
        output_ids = model_outputs['output_ids']
        answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=clean_up_tokenization_spaces)
        conversation = model_outputs['conversation']
        conversation.add_message({'role': 'assistant', 'content': answer})
        return conversation