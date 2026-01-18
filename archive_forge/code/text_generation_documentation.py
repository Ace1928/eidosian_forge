import enum
import warnings
from typing import Dict
from ..utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import Pipeline, build_pipeline_init_args

        Complete the prompt(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                One or several prompts (or one list of prompts) to complete.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to return the tensors of predictions (as token indices) in the outputs. If set to
                `True`, the decoded text is not returned.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to return the decoded texts in the outputs.
            return_full_text (`bool`, *optional*, defaults to `True`):
                If set to `False` only added text is returned, otherwise the full text is returned. Only meaningful if
                *return_text* is set to True.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `True`):
                Whether or not to clean up the potential extra spaces in the text output.
            prefix (`str`, *optional*):
                Prefix added to prompt.
            handle_long_generation (`str`, *optional*):
                By default, this pipelines does not handle long generation (ones that exceed in one form or the other
                the model maximum length). There is no perfect way to adress this (more info
                :https://github.com/huggingface/transformers/issues/14033#issuecomment-948385227). This provides common
                strategies to work around that problem depending on your use case.

                - `None` : default strategy where nothing in particular happens
                - `"hole"`: Truncates left of input, and leaves a gap wide enough to let generation happen (might
                  truncate a lot of the prompt and not suitable when generation exceed the model capacity)

            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Return:
            A list or a list of list of `dict`: Returns one of the following dictionaries (cannot return a combination
            of both `generated_text` and `generated_token_ids`):

            - **generated_text** (`str`, present when `return_text=True`) -- The generated text.
            - **generated_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The token
              ids of the generated text.
        