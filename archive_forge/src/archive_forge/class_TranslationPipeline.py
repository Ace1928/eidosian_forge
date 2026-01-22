import enum
import warnings
from ..tokenization_utils import TruncationStrategy
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
@add_end_docstrings(build_pipeline_init_args(has_tokenizer=True))
class TranslationPipeline(Text2TextGenerationPipeline):
    """
    Translates from one language to another.

    This translation pipeline can currently be loaded from [`pipeline`] using the following task identifier:
    `"translation_xx_to_yy"`.

    The models that this pipeline can use are models that have been fine-tuned on a translation task. See the
    up-to-date list of available models on [huggingface.co/models](https://huggingface.co/models?filter=translation).
    For a list of available parameters, see the [following
    documentation](https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.generation.GenerationMixin.generate)

    Usage:

    ```python
    en_fr_translator = pipeline("translation_en_to_fr")
    en_fr_translator("How old are you?")
    ```"""
    return_name = 'translation'

    def check_inputs(self, input_length: int, min_length: int, max_length: int):
        if input_length > 0.9 * max_length:
            logger.warning(f"Your input_length: {input_length} is bigger than 0.9 * max_length: {max_length}. You might consider increasing your max_length manually, e.g. translator('...', max_length=400)")
        return True

    def preprocess(self, *args, truncation=TruncationStrategy.DO_NOT_TRUNCATE, src_lang=None, tgt_lang=None):
        if getattr(self.tokenizer, '_build_translation_inputs', None):
            return self.tokenizer._build_translation_inputs(*args, return_tensors=self.framework, truncation=truncation, src_lang=src_lang, tgt_lang=tgt_lang)
        else:
            return super()._parse_and_tokenize(*args, truncation=truncation)

    def _sanitize_parameters(self, src_lang=None, tgt_lang=None, **kwargs):
        preprocess_params, forward_params, postprocess_params = super()._sanitize_parameters(**kwargs)
        if src_lang is not None:
            preprocess_params['src_lang'] = src_lang
        if tgt_lang is not None:
            preprocess_params['tgt_lang'] = tgt_lang
        if src_lang is None and tgt_lang is None:
            task = kwargs.get('task', self.task)
            items = task.split('_')
            if task and len(items) == 4:
                preprocess_params['src_lang'] = items[1]
                preprocess_params['tgt_lang'] = items[3]
        return (preprocess_params, forward_params, postprocess_params)

    def __call__(self, *args, **kwargs):
        """
        Translate the text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                Texts to be translated.
            return_tensors (`bool`, *optional*, defaults to `False`):
                Whether or not to include the tensors of predictions (as token indices) in the outputs.
            return_text (`bool`, *optional*, defaults to `True`):
                Whether or not to include the decoded texts in the outputs.
            clean_up_tokenization_spaces (`bool`, *optional*, defaults to `False`):
                Whether or not to clean up the potential extra spaces in the text output.
            src_lang (`str`, *optional*):
                The language of the input. Might be required for multilingual models. Will not have any effect for
                single pair translation models
            tgt_lang (`str`, *optional*):
                The language of the desired output. Might be required for multilingual models. Will not have any effect
                for single pair translation models
            generate_kwargs:
                Additional keyword arguments to pass along to the generate method of the model (see the generate method
                corresponding to your framework [here](./model#generative-models)).

        Return:
            A list or a list of list of `dict`: Each result comes as a dictionary with the following keys:

            - **translation_text** (`str`, present when `return_text=True`) -- The translation.
            - **translation_token_ids** (`torch.Tensor` or `tf.Tensor`, present when `return_tensors=True`) -- The
              token ids of the translation.
        """
        return super().__call__(*args, **kwargs)