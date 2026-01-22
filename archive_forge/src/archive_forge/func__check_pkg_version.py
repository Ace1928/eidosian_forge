from __future__ import annotations
import typing as ty
from types import ModuleType
from packaging.version import Version
from .tripwire import TripWire
def _check_pkg_version(min_version: str | Version) -> ty.Callable[[ModuleType], bool]:
    min_ver = Version(min_version) if isinstance(min_version, str) else min_version

    def check(pkg: ModuleType) -> bool:
        pkg_ver = getattr(pkg, '__version__', None)
        if isinstance(pkg_ver, str):
            return min_ver <= Version(pkg_ver)
        return False
    return check