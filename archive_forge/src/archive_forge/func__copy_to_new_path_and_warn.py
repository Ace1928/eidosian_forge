import warnings
from pathlib import Path
from typing import Optional
from .. import constants
from ._token import get_token
@classmethod
def _copy_to_new_path_and_warn(cls):
    if cls._old_path_token.exists() and (not cls.path_token.exists()):
        cls.save_token(cls._old_path_token.read_text())
        warnings.warn(f'A token has been found in `{cls._old_path_token}`. This is the old path where tokens were stored. The new location is `{cls.path_token}` which is configurable using `HF_HOME` environment variable. Your token has been copied to this new location. You can now safely delete the old token file manually or use `huggingface-cli logout`.')