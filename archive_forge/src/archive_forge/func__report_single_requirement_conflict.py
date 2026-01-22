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
def _report_single_requirement_conflict(self, req: Requirement, parent: Optional[Candidate]) -> DistributionNotFound:
    if parent is None:
        req_disp = str(req)
    else:
        req_disp = f'{req} (from {parent.name})'
    cands = self._finder.find_all_candidates(req.project_name)
    skipped_by_requires_python = self._finder.requires_python_skipped_reasons()
    versions_set: Set[CandidateVersion] = set()
    yanked_versions_set: Set[CandidateVersion] = set()
    for c in cands:
        is_yanked = c.link.is_yanked if c.link else False
        if is_yanked:
            yanked_versions_set.add(c.version)
        else:
            versions_set.add(c.version)
    versions = [str(v) for v in sorted(versions_set)]
    yanked_versions = [str(v) for v in sorted(yanked_versions_set)]
    if yanked_versions:
        logger.critical('Ignored the following yanked versions: %s', ', '.join(yanked_versions) or 'none')
    if skipped_by_requires_python:
        logger.critical('Ignored the following versions that require a different python version: %s', '; '.join(skipped_by_requires_python) or 'none')
    logger.critical('Could not find a version that satisfies the requirement %s (from versions: %s)', req_disp, ', '.join(versions) or 'none')
    if str(req) == 'requirements.txt':
        logger.info('HINT: You are attempting to install a package literally named "requirements.txt" (which cannot exist). Consider using the \'-r\' flag to install the packages listed in requirements.txt')
    return DistributionNotFound(f'No matching distribution found for {req}')