import os
from argparse import Namespace, _SubParsersAction
from functools import wraps
from tempfile import mkstemp
from typing import Any, Callable, Iterable, List, Optional, Union
from ..utils import CachedRepoInfo, CachedRevisionInfo, HFCacheInfo, scan_cache_dir
from . import BaseHuggingfaceCLICommand
from ._cli_utils import ANSI
def _update_expectations(_) -> None:
    checkbox._instruction = _get_expectations_str(hf_cache_info, selected_hashes=[choice['value'] for choice in checkbox.content_control.choices if choice['enabled']])