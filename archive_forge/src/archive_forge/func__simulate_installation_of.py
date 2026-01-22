import logging
from typing import Callable, Dict, List, NamedTuple, Optional, Set, Tuple
from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import LegacySpecifier
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name
from pip._vendor.packaging.version import LegacyVersion
from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.metadata import get_default_environment
from pip._internal.metadata.base import DistributionVersion
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.deprecation import deprecated
def _simulate_installation_of(to_install: List[InstallRequirement], package_set: PackageSet) -> Set[NormalizedName]:
    """Computes the version of packages after installing to_install."""
    installed = set()
    for inst_req in to_install:
        abstract_dist = make_distribution_for_install_requirement(inst_req)
        dist = abstract_dist.get_metadata_distribution()
        name = dist.canonical_name
        package_set[name] = PackageDetails(dist.version, list(dist.iter_dependencies()))
        installed.add(name)
    return installed