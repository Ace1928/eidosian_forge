import logging
import os
from collections.abc import Mapping
from email.headerregistry import Address
from functools import partial, reduce
from inspect import cleandoc
from itertools import chain
from types import MappingProxyType
from typing import (
from ..errors import RemovedConfigError
from ..warnings import SetuptoolsWarning
def _long_description(dist: 'Distribution', val: _DictOrStr, root_dir: _Path):
    from setuptools.config import expand
    if isinstance(val, str):
        file: Union[str, list] = val
        text = expand.read_files(file, root_dir)
        ctype = _guess_content_type(val)
    else:
        file = val.get('file') or []
        text = val.get('text') or expand.read_files(file, root_dir)
        ctype = val['content-type']
    _set_config(dist, 'long_description', text)
    if ctype:
        _set_config(dist, 'long_description_content_type', ctype)
    if file:
        dist._referenced_files.add(cast(str, file))