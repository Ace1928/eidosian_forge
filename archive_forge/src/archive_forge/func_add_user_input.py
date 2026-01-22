import uuid
import warnings
from typing import Any, Dict, List, Union
from ..utils import add_end_docstrings, is_tf_available, is_torch_available, logging
from .base import Pipeline, build_pipeline_init_args
def add_user_input(self, text: str, overwrite: bool=False):
    """
        Add a user input to the conversation for the next round. This is a legacy method that assumes that inputs must
        alternate user/assistant/user/assistant, and so will not add multiple user messages in succession. We recommend
        just using `add_message` with role "user" instead.
        """
    if len(self) > 0 and self[-1]['role'] == 'user':
        if overwrite:
            logger.warning(f'User input added while unprocessed input was existing: "{self[-1]['content']}" was overwritten with: "{text}".')
            self[-1]['content'] = text
        else:
            logger.warning(f'User input added while unprocessed input was existing: "{self[-1]['content']}" new input ignored: "{text}". Set `overwrite` to True to overwrite unprocessed user input')
    else:
        self.messages.append({'role': 'user', 'content': text})