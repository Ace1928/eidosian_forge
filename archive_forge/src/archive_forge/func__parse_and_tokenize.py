import enum
import warnings
from typing import Dict
from ..utils import add_end_docstrings, is_tf_available, is_torch_available
from .base import Pipeline, build_pipeline_init_args
def _parse_and_tokenize(self, *args, **kwargs):
    """
        Parse arguments and tokenize
        """
    if self.model.__class__.__name__ in ['TransfoXLLMHeadModel']:
        kwargs.update({'add_space_before_punct_symbol': True})
    return super()._parse_and_tokenize(*args, **kwargs)