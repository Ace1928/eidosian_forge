import functools
import importlib.metadata
import logging
import os
import pathlib
import sys
import zipfile
import zipimport
from typing import Iterator, List, Optional, Sequence, Set, Tuple
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._internal.metadata.base import BaseDistribution, BaseEnvironment
from pip._internal.models.wheel import Wheel
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filetypes import WHEEL_EXTENSION
from ._compat import BadMetadata, BasePath, get_dist_name, get_info_location
from ._dists import Distribution
@functools.lru_cache(maxsize=None)
def _emit_egg_deprecation(location: Optional[str]) -> None:
    deprecated(reason=f'Loading egg at {location} is deprecated.', replacement='to use pip for package installation.', gone_in='24.3', issue=12330)