from __future__ import annotations
import abc
import os
import typing as T
import re
from .base import ArLikeLinker, RSPFileSyntax
from .. import mesonlib
from ..mesonlib import EnvironmentException, MesonException
from ..arglist import CompilerArgs
class DynamicLinker(metaclass=abc.ABCMeta):
    """Base class for dynamic linkers."""
    _OPTIMIZATION_ARGS: T.Dict[str, T.List[str]] = {'plain': [], '0': [], 'g': [], '1': [], '2': [], '3': [], 's': []}

    @abc.abstractproperty
    def id(self) -> str:
        pass

    def _apply_prefix(self, arg: T.Union[str, T.List[str]]) -> T.List[str]:
        args = [arg] if isinstance(arg, str) else arg
        if self.prefix_arg is None:
            return args
        elif isinstance(self.prefix_arg, str):
            return [self.prefix_arg + arg for arg in args]
        ret: T.List[str] = []
        for arg in args:
            ret += self.prefix_arg + [arg]
        return ret

    def __init__(self, exelist: T.List[str], for_machine: mesonlib.MachineChoice, prefix_arg: T.Union[str, T.List[str]], always_args: T.List[str], *, version: str='unknown version'):
        self.exelist = exelist
        self.for_machine = for_machine
        self.version = version
        self.prefix_arg = prefix_arg
        self.always_args = always_args
        self.machine: T.Optional[str] = None

    def __repr__(self) -> str:
        return '<{}: v{} `{}`>'.format(type(self).__name__, self.version, ' '.join(self.exelist))

    def get_id(self) -> str:
        return self.id

    def get_version_string(self) -> str:
        return f'({self.id} {self.version})'

    def get_exelist(self) -> T.List[str]:
        return self.exelist.copy()

    def get_accepts_rsp(self) -> bool:
        return mesonlib.is_windows()

    def rsp_file_syntax(self) -> RSPFileSyntax:
        """The format of the RSP file that this compiler supports.

        If `self.can_linker_accept_rsp()` returns True, then this needs to
        be implemented
        """
        return RSPFileSyntax.GCC

    def get_always_args(self) -> T.List[str]:
        return self.always_args.copy()

    def get_lib_prefix(self) -> str:
        return ''

    def get_option_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        return []

    def has_multi_arguments(self, args: T.List[str], env: 'Environment') -> T.Tuple[bool, bool]:
        raise EnvironmentException(f'Language {self.id} does not support has_multi_link_arguments.')

    def get_debugfile_name(self, targetfile: str) -> T.Optional[str]:
        """Name of debug file written out (see below)"""
        return None

    def get_debugfile_args(self, targetfile: str) -> T.List[str]:
        """Some compilers (MSVC) write debug into a separate file.

        This method takes the target object path and returns a list of
        commands to append to the linker invocation to control where that
        file is written.
        """
        return []

    def get_optimization_link_args(self, optimization_level: str) -> T.List[str]:
        return mesonlib.listify([self._apply_prefix(a) for a in self._OPTIMIZATION_ARGS[optimization_level]])

    def get_std_shared_lib_args(self) -> T.List[str]:
        return []

    def get_std_shared_module_args(self, options: 'KeyedOptionDictType') -> T.List[str]:
        return self.get_std_shared_lib_args()

    def get_pie_args(self) -> T.List[str]:
        raise EnvironmentException(f'Linker {self.id} does not support position-independent executable')

    def get_lto_args(self) -> T.List[str]:
        return []

    def get_thinlto_cache_args(self, path: str) -> T.List[str]:
        return []

    def sanitizer_args(self, value: str) -> T.List[str]:
        return []

    def get_asneeded_args(self) -> T.List[str]:
        return []

    def get_link_whole_for(self, args: T.List[str]) -> T.List[str]:
        raise EnvironmentException(f'Linker {self.id} does not support link_whole')

    def get_allow_undefined_args(self) -> T.List[str]:
        raise EnvironmentException(f'Linker {self.id} does not support allow undefined')

    @abc.abstractmethod
    def get_output_args(self, outputname: str) -> T.List[str]:
        pass

    def get_coverage_args(self) -> T.List[str]:
        raise EnvironmentException(f"Linker {self.id} doesn't implement coverage data generation.")

    @abc.abstractmethod
    def get_search_args(self, dirname: str) -> T.List[str]:
        pass

    def export_dynamic_args(self, env: 'Environment') -> T.List[str]:
        return []

    def import_library_args(self, implibname: str) -> T.List[str]:
        """The name of the outputted import library.

        This implementation is used only on Windows by compilers that use GNU ld
        """
        return []

    def thread_flags(self, env: 'Environment') -> T.List[str]:
        return []

    def no_undefined_args(self) -> T.List[str]:
        """Arguments to error if there are any undefined symbols at link time.

        This is the inverse of get_allow_undefined_args().

        TODO: A future cleanup might merge this and
              get_allow_undefined_args() into a single method taking a
              boolean
        """
        return []

    def fatal_warnings(self) -> T.List[str]:
        """Arguments to make all warnings errors."""
        return []

    def headerpad_args(self) -> T.List[str]:
        return []

    def get_win_subsystem_args(self, value: str) -> T.List[str]:
        return []

    def bitcode_args(self) -> T.List[str]:
        raise MesonException('This linker does not support bitcode bundles')

    def build_rpath_args(self, env: 'Environment', build_dir: str, from_dir: str, rpath_paths: T.Tuple[str, ...], build_rpath: str, install_rpath: str) -> T.Tuple[T.List[str], T.Set[bytes]]:
        return ([], set())

    def get_soname_args(self, env: 'Environment', prefix: str, shlib_name: str, suffix: str, soversion: str, darwin_versions: T.Tuple[str, str]) -> T.List[str]:
        return []

    def get_archive_name(self, filename: str) -> str:
        return str()

    def get_command_to_archive_shlib(self) -> T.List[str]:
        return []