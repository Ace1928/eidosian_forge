import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
def append_response(self, response: str):
    """
        This is a legacy method. We recommend just using `add_message` with an appropriate role instead.
        """
    self.messages.append({'role': 'assistant', 'content': response})