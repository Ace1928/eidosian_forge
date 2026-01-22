from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, InitVar
from functools import lru_cache
from itertools import chain
from pathlib import Path
import copy
import enum
import json
import os
import pickle
import re
import shlex
import shutil
import typing as T
import hashlib
from .. import build
from .. import dependencies
from .. import programs
from .. import mesonlib
from .. import mlog
from ..compilers import LANGUAGES_USING_LDFLAGS, detect
from ..mesonlib import (
def as_meson_exe_cmdline(self, exe: T.Union[str, mesonlib.File, build.BuildTarget, build.CustomTarget, programs.ExternalProgram], cmd_args: T.Sequence[T.Union[str, mesonlib.File, build.BuildTarget, build.CustomTarget, programs.ExternalProgram]], workdir: T.Optional[str]=None, extra_bdeps: T.Optional[T.List[build.BuildTarget]]=None, capture: T.Optional[str]=None, feed: T.Optional[str]=None, force_serialize: bool=False, env: T.Optional[mesonlib.EnvironmentVariables]=None, verbose: bool=False) -> T.Tuple[T.Sequence[T.Union[str, File, build.Target, programs.ExternalProgram]], str]:
    """
        Serialize an executable for running with a generator or a custom target
        """
    cmd: T.List[T.Union[str, mesonlib.File, build.BuildTarget, build.CustomTarget, programs.ExternalProgram]] = []
    cmd.append(exe)
    cmd.extend(cmd_args)
    es = self.get_executable_serialisation(cmd, workdir, extra_bdeps, capture, feed, env, verbose=verbose)
    reasons: T.List[str] = []
    if es.extra_paths:
        reasons.append('to set PATH')
    if es.exe_wrapper:
        reasons.append('to use exe_wrapper')
    if workdir:
        reasons.append('to set workdir')
    if any(('\n' in c for c in es.cmd_args)):
        reasons.append('because command contains newlines')
    if env and env.varnames:
        reasons.append('to set env')
    can_use_env = not force_serialize
    force_serialize = force_serialize or bool(reasons)
    if capture:
        reasons.append('to capture output')
    if feed:
        reasons.append('to feed input')
    if can_use_env and reasons == ['to set env'] and shutil.which('env'):
        envlist = []
        for k, v in env.get_env({}).items():
            envlist.append(f'{k}={v}')
        return (['env'] + envlist + es.cmd_args, ', '.join(reasons))
    if not force_serialize:
        if not capture and (not feed):
            return (es.cmd_args, '')
        args: T.List[str] = []
        if capture:
            args += ['--capture', capture]
        if feed:
            args += ['--feed', feed]
        return (self.environment.get_build_command() + ['--internal', 'exe'] + args + ['--'] + es.cmd_args, ', '.join(reasons))
    if isinstance(exe, (programs.ExternalProgram, build.BuildTarget, build.CustomTarget)):
        basename = os.path.basename(exe.name)
    elif isinstance(exe, mesonlib.File):
        basename = os.path.basename(exe.fname)
    else:
        basename = os.path.basename(exe)
    hasher = hashlib.sha1()
    if es.env:
        es.env.hash(hasher)
    hasher.update(bytes(str(es.cmd_args), encoding='utf-8'))
    hasher.update(bytes(str(es.workdir), encoding='utf-8'))
    hasher.update(bytes(str(capture), encoding='utf-8'))
    hasher.update(bytes(str(feed), encoding='utf-8'))
    digest = hasher.hexdigest()
    scratch_file = f'meson_exe_{basename}_{digest}.dat'
    exe_data = os.path.join(self.environment.get_scratch_dir(), scratch_file)
    with open(exe_data, 'wb') as f:
        pickle.dump(es, f)
    return (self.environment.get_build_command() + ['--internal', 'exe', '--unpickle', exe_data], ', '.join(reasons))