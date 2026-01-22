import pathlib
import sys
import urllib
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
from ray.data._internal.util import _resolve_custom_scheme
def _is_url(path) -> bool:
    return urllib.parse.urlparse(path).scheme != ''