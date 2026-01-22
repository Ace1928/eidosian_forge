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
def _iter_distributions(self) -> Iterator[BaseDistribution]:
    finder = _DistributionFinder()
    for location in self._paths:
        yield from finder.find(location)
        for dist in finder.find_eggs(location):
            _emit_egg_deprecation(dist.location)
            yield dist
        yield from finder.find_linked(location)