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
def _make_base_candidate_from_link(self, link: Link, template: InstallRequirement, name: Optional[NormalizedName], version: Optional[CandidateVersion]) -> Optional[BaseCandidate]:
    if link in self._build_failures:
        return None
    if template.editable:
        if link not in self._editable_candidate_cache:
            try:
                self._editable_candidate_cache[link] = EditableCandidate(link, template, factory=self, name=name, version=version)
            except MetadataInconsistent as e:
                logger.info('Discarding [blue underline]%s[/]: [yellow]%s[reset]', link, e, extra={'markup': True})
                self._build_failures[link] = e
                return None
        return self._editable_candidate_cache[link]
    else:
        if link not in self._link_candidate_cache:
            try:
                self._link_candidate_cache[link] = LinkCandidate(link, template, factory=self, name=name, version=version)
            except MetadataInconsistent as e:
                logger.info('Discarding [blue underline]%s[/]: [yellow]%s[reset]', link, e, extra={'markup': True})
                self._build_failures[link] = e
                return None
        return self._link_candidate_cache[link]