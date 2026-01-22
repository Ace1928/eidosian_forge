from __future__ import annotations
import copy
from . import mlog, mparser
import pickle, os, uuid
import sys
from itertools import chain
from pathlib import PurePath
from collections import OrderedDict, abc
from dataclasses import dataclass
from .mesonlib import (
from .wrap import WrapMode
import ast
import argparse
import configparser
import enum
import shlex
import typing as T
class CoreData:

    def __init__(self, options: SharedCMDOptions, scratch_dir: str, meson_command: T.List[str]):
        self.lang_guids = {'default': '8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942', 'c': '8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942', 'cpp': '8BC9CEB8-8B4A-11D0-8D11-00A0C91BC942', 'test': '3AC096D0-A1C2-E12C-1390-A8335801FDAB', 'directory': '2150E333-8FDC-42A3-9474-1A3956D46DE8'}
        self.test_guid = str(uuid.uuid4()).upper()
        self.regen_guid = str(uuid.uuid4()).upper()
        self.install_guid = str(uuid.uuid4()).upper()
        self.meson_command = meson_command
        self.target_guids = {}
        self.version = version
        self.options: 'MutableKeyedOptionDictType' = {}
        self.cross_files = self.__load_config_files(options, scratch_dir, 'cross')
        self.compilers: PerMachine[T.Dict[str, Compiler]] = PerMachine(OrderedDict(), OrderedDict())
        self.initialized_subprojects: T.Set[str] = set()
        self.deps: PerMachine[DependencyCache] = PerMachineDefaultable.default(self.is_cross_build(), DependencyCache(self.options, MachineChoice.BUILD), DependencyCache(self.options, MachineChoice.HOST))
        self.compiler_check_cache: T.Dict['CompilerCheckCacheKey', 'CompileResult'] = OrderedDict()
        self.run_check_cache: T.Dict['RunCheckCacheKey', 'RunResult'] = OrderedDict()
        self.cmake_cache: PerMachine[CMakeStateCache] = PerMachine(CMakeStateCache(), CMakeStateCache())
        self.config_files = self.__load_config_files(options, scratch_dir, 'native')
        self.builtin_options_libdir_cross_fixup()
        self.init_builtins('')

    @staticmethod
    def __load_config_files(options: SharedCMDOptions, scratch_dir: str, ftype: str) -> T.List[str]:
        if ftype == 'cross':
            filenames = options.cross_file
        else:
            filenames = options.native_file
        if not filenames:
            return []
        found_invalid: T.List[str] = []
        missing: T.List[str] = []
        real: T.List[str] = []
        for i, f in enumerate(filenames):
            f = os.path.expanduser(os.path.expandvars(f))
            if os.path.exists(f):
                if os.path.isfile(f):
                    real.append(os.path.abspath(f))
                    continue
                elif os.path.isdir(f):
                    found_invalid.append(os.path.abspath(f))
                else:
                    copy = os.path.join(scratch_dir, f'{uuid.uuid4()}.{ftype}.ini')
                    with open(f, encoding='utf-8') as rf:
                        with open(copy, 'w', encoding='utf-8') as wf:
                            wf.write(rf.read())
                    real.append(copy)
                    filenames[i] = copy
                    continue
            if sys.platform != 'win32':
                paths = [os.environ.get('XDG_DATA_HOME', os.path.expanduser('~/.local/share'))] + os.environ.get('XDG_DATA_DIRS', '/usr/local/share:/usr/share').split(':')
                for path in paths:
                    path_to_try = os.path.join(path, 'meson', ftype, f)
                    if os.path.isfile(path_to_try):
                        real.append(path_to_try)
                        break
                else:
                    missing.append(f)
            else:
                missing.append(f)
        if missing:
            if found_invalid:
                mlog.log('Found invalid candidates for', ftype, 'file:', *found_invalid)
            mlog.log('Could not find any valid candidate for', ftype, 'files:', *missing)
            raise MesonException(f'Cannot find specified {ftype} file: {f}')
        return real

    def builtin_options_libdir_cross_fixup(self) -> None:
        if self.cross_files:
            BUILTIN_OPTIONS[OptionKey('libdir')].default = 'lib'

    def sanitize_prefix(self, prefix: str) -> str:
        prefix = os.path.expanduser(prefix)
        if not os.path.isabs(prefix):
            raise MesonException(f'prefix value {prefix!r} must be an absolute path')
        if prefix.endswith('/') or prefix.endswith('\\'):
            if len(prefix) == 3 and prefix[1] == ':':
                pass
            elif len(prefix) == 1:
                pass
            else:
                prefix = prefix[:-1]
        return prefix

    def sanitize_dir_option_value(self, prefix: str, option: OptionKey, value: T.Any) -> T.Any:
        """
        If the option is an installation directory option, the value is an
        absolute path and resides within prefix, return the value
        as a path relative to the prefix. Otherwise, return it as is.

        This way everyone can do f.ex, get_option('libdir') and usually get
        the library directory relative to prefix, even though it really
        should not be relied upon.
        """
        try:
            value = PurePath(value)
        except TypeError:
            return value
        if option.name.endswith('dir') and value.is_absolute() and (option not in BUILTIN_DIR_NOPREFIX_OPTIONS):
            try:
                value = value.relative_to(prefix)
            except ValueError:
                pass
            if '..' in value.parts:
                raise MesonException(f"The value of the '{option}' option is '{value}' but directory options are not allowed to contain '..'.\nIf you need a path outside of the {prefix!r} prefix, please use an absolute path.")
        return value.as_posix()

    def init_builtins(self, subproject: str) -> None:
        for key, opt in BUILTIN_OPTIONS.items():
            self.add_builtin_option(self.options, key.evolve(subproject=subproject), opt)
        for for_machine in iter(MachineChoice):
            for key, opt in BUILTIN_OPTIONS_PER_MACHINE.items():
                self.add_builtin_option(self.options, key.evolve(subproject=subproject, machine=for_machine), opt)

    @staticmethod
    def add_builtin_option(opts_map: 'MutableKeyedOptionDictType', key: OptionKey, opt: 'BuiltinOption') -> None:
        if key.subproject:
            if opt.yielding:
                return
            value = opts_map[key.as_root()].value
        else:
            value = None
        opts_map[key] = opt.init_option(key, value, default_prefix())

    def init_backend_options(self, backend_name: str) -> None:
        if backend_name == 'ninja':
            self.options[OptionKey('backend_max_links')] = UserIntegerOption('Maximum number of linker processes to run or 0 for no limit', (0, None, 0))
        elif backend_name.startswith('vs'):
            self.options[OptionKey('backend_startup_project')] = UserStringOption('Default project to execute in Visual Studio', '')

    def get_option(self, key: OptionKey) -> T.Union[T.List[str], str, int, bool, WrapMode]:
        try:
            v = self.options[key].value
            if key.name == 'wrap_mode':
                return WrapMode[v]
            return v
        except KeyError:
            pass
        try:
            v = self.options[key.as_root()]
            if v.yielding:
                if key.name == 'wrap_mode':
                    return WrapMode[v.value]
                return v.value
        except KeyError:
            pass
        raise MesonException(f'Tried to get unknown builtin option {str(key)}')

    def set_option(self, key: OptionKey, value, first_invocation: bool=False) -> bool:
        dirty = False
        if key.is_builtin():
            if key.name == 'prefix':
                value = self.sanitize_prefix(value)
            else:
                prefix = self.options[OptionKey('prefix')].value
                value = self.sanitize_dir_option_value(prefix, key, value)
        try:
            opt = self.options[key]
        except KeyError:
            raise MesonException(f'Tried to set unknown builtin option {str(key)}')
        if opt.deprecated is True:
            mlog.deprecation(f'Option {key.name!r} is deprecated')
        elif isinstance(opt.deprecated, list):
            for v in opt.listify(value):
                if v in opt.deprecated:
                    mlog.deprecation(f'Option {key.name!r} value {v!r} is deprecated')
        elif isinstance(opt.deprecated, dict):

            def replace(v):
                newvalue = opt.deprecated.get(v)
                if newvalue is not None:
                    mlog.deprecation(f'Option {key.name!r} value {v!r} is replaced by {newvalue!r}')
                    return newvalue
                return v
            newvalue = [replace(v) for v in opt.listify(value)]
            value = ','.join(newvalue)
        elif isinstance(opt.deprecated, str):
            newname = opt.deprecated
            newkey = OptionKey.from_string(newname).evolve(subproject=key.subproject)
            mlog.deprecation(f'Option {key.name!r} is replaced by {newname!r}')
            dirty |= self.set_option(newkey, value, first_invocation)
        changed = opt.set_value(value)
        if changed and opt.readonly and (not first_invocation):
            raise MesonException(f'Tried modify read only option {str(key)!r}')
        dirty |= changed
        if key.name == 'buildtype':
            dirty |= self._set_others_from_buildtype(value)
        return dirty

    def clear_cache(self) -> None:
        self.deps.host.clear()
        self.deps.build.clear()
        self.compiler_check_cache.clear()
        self.run_check_cache.clear()

    def get_nondefault_buildtype_args(self) -> T.List[T.Union[T.Tuple[str, str, str], T.Tuple[str, bool, bool]]]:
        result: T.List[T.Union[T.Tuple[str, str, str], T.Tuple[str, bool, bool]]] = []
        value = self.options[OptionKey('buildtype')].value
        if value == 'plain':
            opt = 'plain'
            debug = False
        elif value == 'debug':
            opt = '0'
            debug = True
        elif value == 'debugoptimized':
            opt = '2'
            debug = True
        elif value == 'release':
            opt = '3'
            debug = False
        elif value == 'minsize':
            opt = 's'
            debug = True
        else:
            assert value == 'custom'
            return []
        actual_opt = self.options[OptionKey('optimization')].value
        actual_debug = self.options[OptionKey('debug')].value
        if actual_opt != opt:
            result.append(('optimization', actual_opt, opt))
        if actual_debug != debug:
            result.append(('debug', actual_debug, debug))
        return result

    def _set_others_from_buildtype(self, value: str) -> bool:
        dirty = False
        if value == 'plain':
            opt = 'plain'
            debug = False
        elif value == 'debug':
            opt = '0'
            debug = True
        elif value == 'debugoptimized':
            opt = '2'
            debug = True
        elif value == 'release':
            opt = '3'
            debug = False
        elif value == 'minsize':
            opt = 's'
            debug = True
        else:
            assert value == 'custom'
            return False
        dirty |= self.options[OptionKey('optimization')].set_value(opt)
        dirty |= self.options[OptionKey('debug')].set_value(debug)
        return dirty

    @staticmethod
    def is_per_machine_option(optname: OptionKey) -> bool:
        if optname.name in BUILTIN_OPTIONS_PER_MACHINE:
            return True
        return optname.lang is not None

    def get_external_args(self, for_machine: MachineChoice, lang: str) -> T.List[str]:
        return T.cast('T.List[str]', self.options[OptionKey('args', machine=for_machine, lang=lang)].value)

    def get_external_link_args(self, for_machine: MachineChoice, lang: str) -> T.List[str]:
        return T.cast('T.List[str]', self.options[OptionKey('link_args', machine=for_machine, lang=lang)].value)

    def update_project_options(self, options: 'MutableKeyedOptionDictType') -> None:
        for key, value in options.items():
            if not key.is_project():
                continue
            if key not in self.options:
                self.options[key] = value
                continue
            oldval = self.options[key]
            if type(oldval) is not type(value):
                self.options[key] = value
            elif oldval.choices != value.choices:
                self.options[key] = value
                try:
                    value.set_value(oldval.value)
                except MesonException:
                    mlog.warning(f'Old value(s) of {key} are no longer valid, resetting to default ({value.value}).', fatal=False)

    def is_cross_build(self, when_building_for: MachineChoice=MachineChoice.HOST) -> bool:
        if when_building_for == MachineChoice.BUILD:
            return False
        return len(self.cross_files) > 0

    def copy_build_options_from_regular_ones(self) -> bool:
        dirty = False
        assert not self.is_cross_build()
        for k in BUILTIN_OPTIONS_PER_MACHINE:
            o = self.options[k]
            dirty |= self.options[k.as_build()].set_value(o.value)
        for bk, bv in self.options.items():
            if bk.machine is MachineChoice.BUILD:
                hk = bk.as_host()
                try:
                    hv = self.options[hk]
                    dirty |= bv.set_value(hv.value)
                except KeyError:
                    continue
        return dirty

    def set_options(self, options: T.Dict[OptionKey, T.Any], subproject: str='', first_invocation: bool=False) -> bool:
        dirty = False
        if not self.is_cross_build():
            options = {k: v for k, v in options.items() if k.machine is not MachineChoice.BUILD}
        pfk = OptionKey('prefix')
        if pfk in options:
            prefix = self.sanitize_prefix(options[pfk])
            dirty |= self.options[OptionKey('prefix')].set_value(prefix)
            for key in BUILTIN_DIR_NOPREFIX_OPTIONS:
                if key not in options:
                    dirty |= self.options[key].set_value(BUILTIN_OPTIONS[key].prefixed_default(key, prefix))
        unknown_options: T.List[OptionKey] = []
        for k, v in options.items():
            if k == pfk:
                continue
            elif k in self.options:
                dirty |= self.set_option(k, v, first_invocation)
            elif k.machine != MachineChoice.BUILD and k.type != OptionType.COMPILER:
                unknown_options.append(k)
        if unknown_options:
            unknown_options_str = ', '.join(sorted((str(s) for s in unknown_options)))
            sub = f'In subproject {subproject}: ' if subproject else ''
            raise MesonException(f'{sub}Unknown options: "{unknown_options_str}"')
        if not self.is_cross_build():
            dirty |= self.copy_build_options_from_regular_ones()
        return dirty

    def set_default_options(self, default_options: T.MutableMapping[OptionKey, str], subproject: str, env: 'Environment') -> None:
        from .compilers import base_options
        options: T.MutableMapping[OptionKey, T.Any] = OrderedDict()
        for k, v in default_options.items():
            if not subproject or k.subproject == subproject:
                options[k] = v
        options.update(env.options)
        env.options = options
        options = OrderedDict()
        for k, v in env.options.items():
            if k.subproject and k.subproject != subproject:
                continue
            if subproject and k.is_builtin() and self.options[k.evolve(subproject='', machine=MachineChoice.HOST)].yielding:
                continue
            if k.type in {OptionType.COMPILER, OptionType.BACKEND}:
                continue
            if k.type == OptionType.BASE and k.as_root() in base_options:
                continue
            options[k] = v
        self.set_options(options, subproject=subproject, first_invocation=env.first_invocation)

    def add_compiler_options(self, options: MutableKeyedOptionDictType, lang: str, for_machine: MachineChoice, env: Environment, subproject: str) -> None:
        for k, o in options.items():
            value = env.options.get(k)
            if value is not None:
                o.set_value(value)
                if not subproject:
                    self.options[k] = o
            self.options.setdefault(k, o)
            if subproject:
                sk = k.evolve(subproject=subproject)
                value = env.options.get(sk) or value
                if value is not None:
                    o.set_value(value)
                    self.options[sk] = o
                self.options.setdefault(sk, o)

    def add_lang_args(self, lang: str, comp: T.Type['Compiler'], for_machine: MachineChoice, env: 'Environment') -> None:
        """Add global language arguments that are needed before compiler/linker detection."""
        from .compilers import compilers
        self.options.update(compilers.get_global_options(lang, comp, for_machine, env))

    def process_compiler_options(self, lang: str, comp: Compiler, env: Environment, subproject: str) -> None:
        from . import compilers
        self.add_compiler_options(comp.get_options(), lang, comp.for_machine, env, subproject)
        enabled_opts: T.List[OptionKey] = []
        for key in comp.base_options:
            if subproject:
                skey = key.evolve(subproject=subproject)
            else:
                skey = key
            if skey not in self.options:
                self.options[skey] = copy.deepcopy(compilers.base_options[key])
                if skey in env.options:
                    self.options[skey].set_value(env.options[skey])
                    enabled_opts.append(skey)
                elif subproject and key in env.options:
                    self.options[skey].set_value(env.options[key])
                    enabled_opts.append(skey)
                if subproject and key not in self.options:
                    self.options[key] = copy.deepcopy(self.options[skey])
            elif skey in env.options:
                self.options[skey].set_value(env.options[skey])
            elif subproject and key in env.options:
                self.options[skey].set_value(env.options[key])
        self.emit_base_options_warnings(enabled_opts)

    def emit_base_options_warnings(self, enabled_opts: T.List[OptionKey]) -> None:
        if OptionKey('b_bitcode') in enabled_opts:
            mlog.warning("Base option 'b_bitcode' is enabled, which is incompatible with many linker options. Incompatible options such as 'b_asneeded' have been disabled.", fatal=False)
            mlog.warning('Please see https://mesonbuild.com/Builtin-options.html#Notes_about_Apple_Bitcode_support for more details.', fatal=False)