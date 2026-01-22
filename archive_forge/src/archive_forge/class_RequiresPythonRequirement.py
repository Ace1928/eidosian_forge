from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._internal.req.constructors import install_req_drop_extras
from pip._internal.req.req_install import InstallRequirement
from .base import Candidate, CandidateLookup, Requirement, format_name
class RequiresPythonRequirement(Requirement):
    """A requirement representing Requires-Python metadata."""

    def __init__(self, specifier: SpecifierSet, match: Candidate) -> None:
        self.specifier = specifier
        self._candidate = match

    def __str__(self) -> str:
        return f'Python {self.specifier}'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({str(self.specifier)!r})'

    @property
    def project_name(self) -> NormalizedName:
        return self._candidate.project_name

    @property
    def name(self) -> str:
        return self._candidate.name

    def format_for_error(self) -> str:
        return str(self)

    def get_candidate_lookup(self) -> CandidateLookup:
        if self.specifier.contains(self._candidate.version, prereleases=True):
            return (self._candidate, None)
        return (None, None)

    def is_satisfied_by(self, candidate: Candidate) -> bool:
        assert candidate.name == self._candidate.name, 'Not Python candidate'
        return self.specifier.contains(candidate.version, prereleases=True)