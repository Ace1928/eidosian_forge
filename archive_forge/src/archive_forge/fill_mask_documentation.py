from typing import Dict
import numpy as np
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import GenericTensor, Pipeline, PipelineException, build_pipeline_init_args

        Fill the masked token in the text(s) given as inputs.

        Args:
            args (`str` or `List[str]`):
                One or several texts (or one list of prompts) with masked tokens.
            targets (`str` or `List[str]`, *optional*):
                When passed, the model will limit the scores to the passed targets instead of looking up in the whole
                vocab. If the provided targets are not in the model vocab, they will be tokenized and the first
                resulting token will be used (with a warning, and that might be slower).
            top_k (`int`, *optional*):
                When passed, overrides the number of predictions to return.

        Return:
            A list or a list of list of `dict`: Each result comes as list of dictionaries with the following keys:

            - **sequence** (`str`) -- The corresponding input with the mask token prediction.
            - **score** (`float`) -- The corresponding probability.
            - **token** (`int`) -- The predicted token id (to replace the masked one).
            - **token_str** (`str`) -- The predicted token (to replace the masked one).
        