import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
@property
def _user_messages(self):
    return [message['content'] for message in self.messages if message['role'] == 'user']