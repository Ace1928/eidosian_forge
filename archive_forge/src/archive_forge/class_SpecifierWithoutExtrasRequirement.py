from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._internal.req.constructors import install_req_drop_extras
from pip._internal.req.req_install import InstallRequirement
from .base import Candidate, CandidateLookup, Requirement, format_name
class SpecifierWithoutExtrasRequirement(SpecifierRequirement):
    """
    Requirement backed by an install requirement on a base package.
    Trims extras from its install requirement if there are any.
    """

    def __init__(self, ireq: InstallRequirement) -> None:
        assert ireq.link is None, 'This is a link, not a specifier'
        self._ireq = install_req_drop_extras(ireq)
        self._extras = frozenset((canonicalize_name(e) for e in self._ireq.extras))