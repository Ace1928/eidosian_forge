import io
import logging
import os
import pathlib
import shutil
import sys
import tempfile
from collections import OrderedDict
from contextlib import contextmanager
from typing import (IO, Dict, Iterable, Iterator, Mapping, Optional, Tuple,
from .parser import Binding, parse_stream
from .variables import parse_variables
def unset_key(dotenv_path: StrPath, key_to_unset: str, quote_mode: str='always', encoding: Optional[str]='utf-8') -> Tuple[Optional[bool], str]:
    """
    Removes a given key from the given `.env` file.

    If the .env path given doesn't exist, fails.
    If the given key doesn't exist in the .env, fails.
    """
    if not os.path.exists(dotenv_path):
        logger.warning("Can't delete from %s - it doesn't exist.", dotenv_path)
        return (None, key_to_unset)
    removed = False
    with rewrite(dotenv_path, encoding=encoding) as (source, dest):
        for mapping in with_warn_for_invalid_lines(parse_stream(source)):
            if mapping.key == key_to_unset:
                removed = True
            else:
                dest.write(mapping.original.string)
    if not removed:
        logger.warning("Key %s not removed from %s - key doesn't exist.", key_to_unset, dotenv_path)
        return (None, key_to_unset)
    return (removed, key_to_unset)