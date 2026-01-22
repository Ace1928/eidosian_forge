from __future__ import annotations
import re
import os, os.path, pathlib
import shutil
import typing as T
from . import ExtensionModule, ModuleReturnValue, ModuleObject, ModuleInfo
from .. import build, mesonlib, mlog, dependencies
from ..cmake import TargetOptions, cmake_defines_to_args
from ..interpreter import SubprojectHolder
from ..interpreter.type_checking import REQUIRED_KW, INSTALL_DIR_KW, NoneType, in_set_validator
from ..interpreterbase import (
class CMakeSubproject(ModuleObject):

    def __init__(self, subp: SubprojectHolder):
        assert isinstance(subp, SubprojectHolder)
        assert subp.cm_interpreter is not None
        super().__init__()
        self.subp = subp
        self.cm_interpreter = subp.cm_interpreter
        self.methods.update({'get_variable': self.get_variable, 'dependency': self.dependency, 'include_directories': self.include_directories, 'target': self.target, 'target_type': self.target_type, 'target_list': self.target_list, 'found': self.found_method})

    def _args_to_info(self, args: T.List[str]) -> T.Dict[str, str]:
        if len(args) != 1:
            raise InterpreterException('Exactly one argument is required.')
        tgt = args[0]
        res = self.cm_interpreter.target_info(tgt)
        if res is None:
            raise InterpreterException(f'The CMake target {tgt} does not exist\n' + '  Use the following command in your meson.build to list all available targets:\n\n' + "    message('CMake targets:\\n - ' + '\\n - '.join(<cmake_subproject>.target_list()))")
        assert all((x in res for x in ['inc', 'src', 'dep', 'tgt', 'func']))
        return res

    @noKwargs
    @stringArgs
    def get_variable(self, state: ModuleState, args: T.List[str], kwargs: TYPE_kwargs) -> TYPE_var:
        return self.subp.get_variable_method(args, kwargs)

    @FeatureNewKwargs('dependency', '0.56.0', ['include_type'])
    @permittedKwargs({'include_type'})
    @stringArgs
    def dependency(self, state: ModuleState, args: T.List[str], kwargs: T.Dict[str, str]) -> dependencies.Dependency:
        info = self._args_to_info(args)
        if info['func'] == 'executable':
            raise InvalidArguments(f'{args[0]} is an executable and does not support the dependency() method. Use target() instead.')
        orig = self.get_variable(state, [info['dep']], {})
        assert isinstance(orig, dependencies.Dependency)
        actual = orig.include_type
        if 'include_type' in kwargs and kwargs['include_type'] != actual:
            mlog.debug('Current include type is {}. Converting to requested {}'.format(actual, kwargs['include_type']))
            return orig.generate_system_dependency(kwargs['include_type'])
        return orig

    @noKwargs
    @stringArgs
    def include_directories(self, state: ModuleState, args: T.List[str], kwargs: TYPE_kwargs) -> build.IncludeDirs:
        info = self._args_to_info(args)
        return self.get_variable(state, [info['inc']], kwargs)

    @noKwargs
    @stringArgs
    def target(self, state: ModuleState, args: T.List[str], kwargs: TYPE_kwargs) -> build.Target:
        info = self._args_to_info(args)
        return self.get_variable(state, [info['tgt']], kwargs)

    @noKwargs
    @stringArgs
    def target_type(self, state: ModuleState, args: T.List[str], kwargs: TYPE_kwargs) -> str:
        info = self._args_to_info(args)
        return info['func']

    @noPosargs
    @noKwargs
    def target_list(self, state: ModuleState, args: TYPE_var, kwargs: TYPE_kwargs) -> T.List[str]:
        return self.cm_interpreter.target_list()

    @noPosargs
    @noKwargs
    @FeatureNew('CMakeSubproject.found()', '0.53.2')
    def found_method(self, state: ModuleState, args: TYPE_var, kwargs: TYPE_kwargs) -> bool:
        return self.subp is not None