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
def _find_eggs_in_dir(self, location: str) -> Iterator[BaseDistribution]:
    from pip._vendor.pkg_resources import find_distributions
    from pip._internal.metadata import pkg_resources as legacy
    with os.scandir(location) as it:
        for entry in it:
            if not entry.name.endswith('.egg'):
                continue
            for dist in find_distributions(entry.path):
                yield legacy.Distribution(dist)