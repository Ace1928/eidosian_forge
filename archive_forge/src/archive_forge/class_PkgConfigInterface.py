from __future__ import annotations
from pathlib import Path
from .base import ExternalDependency, DependencyException, sort_libpaths, DependencyTypeName
from ..mesonlib import EnvironmentVariables, OptionKey, OrderedSet, PerMachine, Popen_safe, Popen_safe_logged, MachineChoice, join_args
from ..programs import find_external_program, ExternalProgram
from .. import mlog
from pathlib import PurePath
from functools import lru_cache
import re
import os
import shlex
import typing as T
class PkgConfigInterface:
    """Base class wrapping a pkg-config implementation"""
    class_impl: PerMachine[T.Union[Literal[False], T.Optional[PkgConfigInterface]]] = PerMachine(False, False)
    class_cli_impl: PerMachine[T.Union[Literal[False], T.Optional[PkgConfigCLI]]] = PerMachine(False, False)

    @staticmethod
    def instance(env: Environment, for_machine: MachineChoice, silent: bool) -> T.Optional[PkgConfigInterface]:
        """Return a pkg-config implementation singleton"""
        for_machine = for_machine if env.is_cross_build() else MachineChoice.HOST
        impl = PkgConfigInterface.class_impl[for_machine]
        if impl is False:
            impl = PkgConfigCLI(env, for_machine, silent)
            if not impl.found():
                impl = None
            if not impl and (not silent):
                mlog.log('Found pkg-config:', mlog.red('NO'))
            PkgConfigInterface.class_impl[for_machine] = impl
        return impl

    @staticmethod
    def _cli(env: Environment, for_machine: MachineChoice, silent: bool=False) -> T.Optional[PkgConfigCLI]:
        """Return the CLI pkg-config implementation singleton
        Even when we use another implementation internally, external tools might
        still need the CLI implementation.
        """
        for_machine = for_machine if env.is_cross_build() else MachineChoice.HOST
        impl: T.Union[Literal[False], T.Optional[PkgConfigInterface]]
        impl = PkgConfigInterface.instance(env, for_machine, silent)
        if impl and (not isinstance(impl, PkgConfigCLI)):
            impl = PkgConfigInterface.class_cli_impl[for_machine]
            if impl is False:
                impl = PkgConfigCLI(env, for_machine, silent)
                if not impl.found():
                    impl = None
                PkgConfigInterface.class_cli_impl[for_machine] = impl
        return T.cast('T.Optional[PkgConfigCLI]', impl)

    @staticmethod
    def get_env(env: Environment, for_machine: MachineChoice, uninstalled: bool=False) -> EnvironmentVariables:
        cli = PkgConfigInterface._cli(env, for_machine)
        return cli._get_env(uninstalled) if cli else EnvironmentVariables()

    @staticmethod
    def setup_env(environ: EnvironOrDict, env: Environment, for_machine: MachineChoice, uninstalled: bool=False) -> EnvironOrDict:
        cli = PkgConfigInterface._cli(env, for_machine)
        return cli._setup_env(environ, uninstalled) if cli else environ

    def __init__(self, env: Environment, for_machine: MachineChoice) -> None:
        self.env = env
        self.for_machine = for_machine

    def found(self) -> bool:
        """Return whether pkg-config is supported"""
        raise NotImplementedError

    def version(self, name: str) -> T.Optional[str]:
        """Return module version or None if not found"""
        raise NotImplementedError

    def cflags(self, name: str, allow_system: bool=False, define_variable: PkgConfigDefineType=None) -> ImmutableListProtocol[str]:
        """Return module cflags
           @allow_system: If False, remove default system include paths
        """
        raise NotImplementedError

    def libs(self, name: str, static: bool=False, allow_system: bool=False, define_variable: PkgConfigDefineType=None) -> ImmutableListProtocol[str]:
        """Return module libs
           @static: If True, also include private libraries
           @allow_system: If False, remove default system libraries search paths
        """
        raise NotImplementedError

    def variable(self, name: str, variable_name: str, define_variable: PkgConfigDefineType) -> T.Optional[str]:
        """Return module variable or None if variable is not defined"""
        raise NotImplementedError

    def list_all(self) -> ImmutableListProtocol[str]:
        """Return all available pkg-config modules"""
        raise NotImplementedError