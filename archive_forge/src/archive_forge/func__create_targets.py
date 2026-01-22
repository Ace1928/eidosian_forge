from __future__ import annotations
from pathlib import Path
import os
import shlex
import subprocess
import typing as T
from . import ExtensionModule, ModuleReturnValue, NewExtensionModule, ModuleInfo
from .. import mlog, build
from ..compilers.compilers import CFLAGS_MAPPING
from ..envconfig import ENV_VAR_PROG_MAP
from ..dependencies import InternalDependency
from ..dependencies.pkgconfig import PkgConfigInterface
from ..interpreterbase import FeatureNew
from ..interpreter.type_checking import ENV_KW, DEPENDS_KW
from ..interpreterbase.decorators import ContainerTypeInfo, KwargInfo, typed_kwargs, typed_pos_args
from ..mesonlib import (EnvironmentException, MesonException, Popen_safe, MachineChoice,
def _create_targets(self, extra_depends: T.List[T.Union['BuildTarget', 'CustomTarget']]) -> T.List['TYPE_var']:
    cmd = self.env.get_build_command()
    cmd += ['--internal', 'externalproject', '--name', self.name, '--srcdir', self.src_dir.as_posix(), '--builddir', self.build_dir.as_posix(), '--installdir', self.install_dir.as_posix(), '--logdir', mlog.get_log_dir(), '--make', join_args(self.make)]
    if self.verbose:
        cmd.append('--verbose')
    self.target = build.CustomTarget(self.name, self.subdir.as_posix(), self.subproject, self.env, cmd + ['@OUTPUT@', '@DEPFILE@'], [], [f'{self.name}.stamp'], depfile=f'{self.name}.d', console=True, extra_depends=extra_depends, description='Generating external project {}')
    idir = build.InstallDir(self.subdir.as_posix(), Path('dist', self.rel_prefix).as_posix(), install_dir='.', install_dir_name='.', install_mode=None, exclude=None, strip_directory=True, from_source_dir=False, subproject=self.subproject)
    return [self.target, idir]