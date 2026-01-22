import logging
import sys
from typing import TYPE_CHECKING, Any, FrozenSet, Iterable, Optional, Tuple, Union, cast
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import Version
from pip._internal.exceptions import (
from pip._internal.metadata import BaseDistribution
from pip._internal.models.link import Link, links_equivalent
from pip._internal.models.wheel import Wheel
from pip._internal.req.constructors import (
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.direct_url_helpers import direct_url_from_link
from pip._internal.utils.misc import normalize_version_info
from .base import Candidate, CandidateVersion, Requirement, format_name
class EditableCandidate(_InstallRequirementBackedCandidate):
    is_editable = True

    def __init__(self, link: Link, template: InstallRequirement, factory: 'Factory', name: Optional[NormalizedName]=None, version: Optional[CandidateVersion]=None) -> None:
        super().__init__(link=link, source_link=link, ireq=make_install_req_from_editable(link, template), factory=factory, name=name, version=version)

    def _prepare_distribution(self) -> BaseDistribution:
        return self._factory.preparer.prepare_editable_requirement(self._ireq)