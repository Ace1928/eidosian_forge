import contextlib
import functools
import logging
from typing import (
from pip._vendor.packaging.requirements import InvalidRequirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.resolvelib import ResolutionImpossible
from pip._internal.cache import CacheEntry, WheelCache
from pip._internal.exceptions import (
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import BaseDistribution, get_default_environment
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.operations.prepare import RequirementPreparer
from pip._internal.req.constructors import (
from pip._internal.req.req_install import (
from pip._internal.resolution.base import InstallRequirementProvider
from pip._internal.utils.compatibility_tags import get_supported
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.packaging import get_requirement
from pip._internal.utils.virtualenv import running_under_virtualenv
from .base import Candidate, CandidateVersion, Constraint, Requirement
from .candidates import (
from .found_candidates import FoundCandidates, IndexCandidateInfo
from .requirements import (
def _make_requirements_from_install_req(self, ireq: InstallRequirement, requested_extras: Iterable[str]) -> Iterator[Requirement]:
    """
        Returns requirement objects associated with the given InstallRequirement. In
        most cases this will be a single object but the following special cases exist:
            - the InstallRequirement has markers that do not apply -> result is empty
            - the InstallRequirement has both a constraint (or link) and extras
                -> result is split in two requirement objects: one with the constraint
                (or link) and one with the extra. This allows centralized constraint
                handling for the base, resulting in fewer candidate rejections.
        """
    if not ireq.match_markers(requested_extras):
        logger.info("Ignoring %s: markers '%s' don't match your environment", ireq.name, ireq.markers)
    elif not ireq.link:
        if ireq.extras and ireq.req is not None and ireq.req.specifier:
            yield SpecifierWithoutExtrasRequirement(ireq)
        yield SpecifierRequirement(ireq)
    else:
        self._fail_if_link_is_unsupported_wheel(ireq.link)
        cand = self._make_base_candidate_from_link(ireq.link, template=install_req_drop_extras(ireq) if ireq.extras else ireq, name=canonicalize_name(ireq.name) if ireq.name else None, version=None)
        if cand is None:
            if not ireq.name:
                raise self._build_failures[ireq.link]
            yield UnsatisfiableRequirement(canonicalize_name(ireq.name))
        else:
            yield self.make_requirement_from_candidate(cand)
            if ireq.extras:
                yield self.make_requirement_from_candidate(self._make_extras_candidate(cand, frozenset(ireq.extras)))