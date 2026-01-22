import os
import re
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple, Union
from langchain_core.stores import ByteStore
from langchain.storage.exceptions import InvalidKeyException
def _get_full_path(self, key: str) -> Path:
    """Get the full path for a given key relative to the root path.

        Args:
            key (str): The key relative to the root path.

        Returns:
            Path: The full path for the given key.
        """
    if not re.match('^[a-zA-Z0-9_.\\-/]+$', key):
        raise InvalidKeyException(f'Invalid characters in key: {key}')
    full_path = os.path.abspath(self.root_path / key)
    common_path = os.path.commonpath([str(self.root_path), full_path])
    if common_path != str(self.root_path):
        raise InvalidKeyException(f'Invalid key: {key}. Key should be relative to the full path.{self.root_path} vs. {common_path} and full path of {full_path}')
    return Path(full_path)