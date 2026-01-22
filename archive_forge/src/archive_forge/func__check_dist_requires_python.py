import logging
import sys
from collections import defaultdict
from itertools import chain
from typing import DefaultDict, Iterable, List, Optional, Set, Tuple
from pip._vendor.packaging import specifiers
from pip._vendor.packaging.requirements import Requirement
from pip._internal.cache import WheelCache
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.req_install import (
from pip._internal.req.req_set import RequirementSet
from pip._internal.resolution.base import BaseResolver, InstallRequirementProvider
from pip._internal.utils import compatibility_tags
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import normalize_version_info
from pip._internal.utils.packaging import check_requires_python
def _check_dist_requires_python(dist: BaseDistribution, version_info: Tuple[int, int, int], ignore_requires_python: bool=False) -> None:
    """
    Check whether the given Python version is compatible with a distribution's
    "Requires-Python" value.

    :param version_info: A 3-tuple of ints representing the Python
        major-minor-micro version to check.
    :param ignore_requires_python: Whether to ignore the "Requires-Python"
        value if the given Python version isn't compatible.

    :raises UnsupportedPythonVersion: When the given Python version isn't
        compatible.
    """
    try:
        requires_python = str(dist.requires_python)
    except FileNotFoundError as e:
        raise NoneMetadataError(dist, str(e))
    try:
        is_compatible = check_requires_python(requires_python, version_info=version_info)
    except specifiers.InvalidSpecifier as exc:
        logger.warning('Package %r has an invalid Requires-Python: %s', dist.raw_name, exc)
        return
    if is_compatible:
        return
    version = '.'.join(map(str, version_info))
    if ignore_requires_python:
        logger.debug('Ignoring failed Requires-Python check for package %r: %s not in %r', dist.raw_name, version, requires_python)
        return
    raise UnsupportedPythonVersion('Package {!r} requires a different Python: {} not in {!r}'.format(dist.raw_name, version, requires_python))