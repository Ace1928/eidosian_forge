import io
import logging
import os
import pkgutil
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging import Logger
from typing import IO, Any, Iterable, Iterator, List, Optional, Tuple, Union, cast
from blib2to3.pgen2.grammar import Grammar
from blib2to3.pgen2.tokenize import GoodTokenInfo
from blib2to3.pytree import NL
from . import grammar, parse, pgen, token, tokenize
def _generate_pickle_name(gt: Path, cache_dir: Optional[Path]=None) -> str:
    head, tail = os.path.splitext(gt)
    if tail == '.txt':
        tail = ''
    name = head + tail + '.'.join(map(str, sys.version_info)) + '.pickle'
    if cache_dir:
        return os.path.join(cache_dir, os.path.basename(name))
    else:
        return name