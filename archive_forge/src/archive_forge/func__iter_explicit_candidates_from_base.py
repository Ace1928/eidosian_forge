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
def _iter_explicit_candidates_from_base(self, base_requirements: Iterable[Requirement], extras: FrozenSet[str]) -> Iterator[Candidate]:
    """Produce explicit candidates from the base given an extra-ed package.

        :param base_requirements: Requirements known to the resolver. The
            requirements are guaranteed to not have extras.
        :param extras: The extras to inject into the explicit requirements'
            candidates.
        """
    for req in base_requirements:
        lookup_cand, _ = req.get_candidate_lookup()
        if lookup_cand is None:
            continue
        base_cand = as_base_candidate(lookup_cand)
        assert base_cand is not None, 'no extras here'
        yield self._make_extras_candidate(base_cand, extras)