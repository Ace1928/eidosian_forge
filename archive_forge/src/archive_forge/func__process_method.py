from __future__ import annotations
import functools
import typing as T
from .base import DependencyException, DependencyMethods
from .base import process_method_kw
from .base import BuiltinDependency, SystemDependency
from .cmake import CMakeDependency
from .framework import ExtraFrameworkDependency
from .pkgconfig import PkgConfigDependency
@staticmethod
def _process_method(method: DependencyMethods, env: 'Environment', for_machine: MachineChoice) -> bool:
    """Report whether a method is valid or not.

        If the method is valid, return true, otherwise return false. This is
        used in a list comprehension to filter methods that are not possible.

        By default this only remove EXTRAFRAMEWORK dependencies for non-mac platforms.
        """
    if method is DependencyMethods.EXTRAFRAMEWORK and (not env.machines[for_machine].is_darwin()):
        return False
    return True