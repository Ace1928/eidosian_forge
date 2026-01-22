from __future__ import annotations
import itertools
import os, platform, re, sys, shutil
import typing as T
import collections
from . import coredata
from . import mesonlib
from .mesonlib import (
from . import mlog
from .programs import ExternalProgram
from .envconfig import (
from . import compilers
from .compilers import (
from functools import lru_cache
from mesonbuild import envconfig
def _set_default_options_from_env(self) -> None:
    opts: T.List[T.Tuple[str, str]] = [(v, f'{k}_args') for k, v in compilers.compilers.CFLAGS_MAPPING.items()] + [('PKG_CONFIG_PATH', 'pkg_config_path'), ('CMAKE_PREFIX_PATH', 'cmake_prefix_path'), ('LDFLAGS', 'ldflags'), ('CPPFLAGS', 'cppflags')]
    env_opts: T.DefaultDict[OptionKey, T.List[str]] = collections.defaultdict(list)
    for (evar, keyname), for_machine in itertools.product(opts, MachineChoice):
        p_env = _get_env_var(for_machine, self.is_cross_build(), evar)
        if p_env is not None:
            if keyname == 'cmake_prefix_path':
                if self.machines[for_machine].is_windows():
                    _p_env = p_env.split(os.pathsep)
                else:
                    _p_env = re.split(':|;', p_env)
                p_list = list(mesonlib.OrderedSet(_p_env))
            elif keyname == 'pkg_config_path':
                p_list = list(mesonlib.OrderedSet(p_env.split(os.pathsep)))
            else:
                p_list = split_args(p_env)
            p_list = [e for e in p_list if e]
            if self.first_invocation:
                if keyname == 'ldflags':
                    key = OptionKey('link_args', machine=for_machine, lang='c')
                    for lang in compilers.compilers.LANGUAGES_USING_LDFLAGS:
                        key = key.evolve(lang=lang)
                        env_opts[key].extend(p_list)
                elif keyname == 'cppflags':
                    key = OptionKey('env_args', machine=for_machine, lang='c')
                    for lang in compilers.compilers.LANGUAGES_USING_CPPFLAGS:
                        key = key.evolve(lang=lang)
                        env_opts[key].extend(p_list)
                else:
                    key = OptionKey.from_string(keyname).evolve(machine=for_machine)
                    if evar in compilers.compilers.CFLAGS_MAPPING.values():
                        key = key.evolve('env_args')
                    env_opts[key].extend(p_list)
    for k, v in env_opts.items():
        if k not in self.options:
            self.options[k] = v