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
def _create_whitelist(would_be_installed: Set[NormalizedName], package_set: PackageSet) -> Set[NormalizedName]:
    packages_affected = set(would_be_installed)
    for package_name in package_set:
        if package_name in packages_affected:
            continue
        for req in package_set[package_name].dependencies:
            if canonicalize_name(req.name) in packages_affected:
                packages_affected.add(package_name)
                break
    return packages_affected