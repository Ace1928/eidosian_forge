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
def _get_dist_for(self, req: InstallRequirement) -> BaseDistribution:
    """Takes a InstallRequirement and returns a single AbstractDist         representing a prepared variant of the same.
        """
    if req.editable:
        return self.preparer.prepare_editable_requirement(req)
    assert req.satisfied_by is None
    skip_reason = self._check_skip_installed(req)
    if req.satisfied_by:
        return self.preparer.prepare_installed_requirement(req, skip_reason)
    self._populate_link(req)
    dist = self.preparer.prepare_linked_requirement(req)
    if not self.ignore_installed:
        req.check_if_exists(self.use_user_site)
    if req.satisfied_by:
        should_modify = self.upgrade_strategy != 'to-satisfy-only' or self.force_reinstall or self.ignore_installed or (req.link.scheme == 'file')
        if should_modify:
            self._set_req_to_reinstall(req)
        else:
            logger.info('Requirement already satisfied (use --upgrade to upgrade): %s', req)
    return dist