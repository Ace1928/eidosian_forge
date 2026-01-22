from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from pathlib import PurePath, Path
from textwrap import dedent
import itertools
import json
import os
import pickle
import re
import subprocess
import typing as T
from . import backends
from .. import modules
from .. import environment, mesonlib
from .. import build
from .. import mlog
from .. import compilers
from ..arglist import CompilerArgs
from ..compilers import Compiler
from ..linkers import ArLikeLinker, RSPFileSyntax
from ..mesonlib import (
from ..mesonlib import get_compiler_for_source, has_path_sep, OptionKey
from .backends import CleanTrees
from ..build import GeneratedList, InvalidArguments
class NinjaBackend(backends.Backend):

    def __init__(self, build: T.Optional[build.Build], interpreter: T.Optional[Interpreter]):
        super().__init__(build, interpreter)
        self.name = 'ninja'
        self.ninja_filename = 'build.ninja'
        self.fortran_deps = {}
        self.all_outputs: T.Set[str] = set()
        self.introspection_data = {}
        self.created_llvm_ir_rule = PerMachine(False, False)
        self.rust_crates: T.Dict[str, RustCrate] = {}
        self.implicit_meson_outs = []

    def create_phony_target(self, dummy_outfile: str, rulename: str, phony_infilename: str) -> NinjaBuildElement:
        """
        We need to use aliases for targets that might be used as directory
        names to workaround a Ninja bug that breaks `ninja -t clean`.
        This is used for 'reserved' targets such as 'test', 'install',
        'benchmark', etc, and also for RunTargets.
        https://github.com/mesonbuild/meson/issues/1644
        """
        if dummy_outfile.startswith('meson-internal__'):
            raise AssertionError(f'Invalid usage of create_phony_target with {dummy_outfile!r}')
        to_name = f'meson-internal__{dummy_outfile}'
        elem = NinjaBuildElement(self.all_outputs, dummy_outfile, 'phony', to_name)
        self.add_build(elem)
        return NinjaBuildElement(self.all_outputs, to_name, rulename, phony_infilename)

    def detect_vs_dep_prefix(self, tempfilename):
        """VS writes its dependency in a locale dependent format.
        Detect the search prefix to use."""
        for compiler in self.environment.coredata.compilers.host.values():
            if compiler.language in {'fortran', 'masm'}:
                continue
            if compiler.id == 'pgi' and mesonlib.is_windows():
                return open(tempfilename, 'a', encoding='utf-8')
            if compiler.get_argument_syntax() == 'msvc':
                break
        else:
            return open(tempfilename, 'a', encoding='utf-8')
        filebase = 'incdetect.' + compilers.lang_suffixes[compiler.language][0]
        filename = os.path.join(self.environment.get_scratch_dir(), filebase)
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(dedent('                #include<stdio.h>\n                int dummy;\n            '))
        pc = subprocess.Popen(compiler.get_exelist() + ['/showIncludes', '/c', filebase], cwd=self.environment.get_scratch_dir(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = pc.communicate()
        matchre = re.compile(b'^(.*\\s)([a-zA-Z]:[\\\\/]|[\\\\\\/]).*stdio.h$')

        def detect_prefix(out):
            for line in re.split(b'\\r?\\n', out):
                match = matchre.match(line)
                if match:
                    with open(tempfilename, 'ab') as binfile:
                        binfile.write(b'msvc_deps_prefix = ' + match.group(1) + b'\n')
                    return open(tempfilename, 'a', encoding='utf-8')
            return None
        result = detect_prefix(stdout) or detect_prefix(stderr)
        if result:
            return result
        raise MesonException(f'Could not determine vs dep dependency prefix string. output: {stderr} {stdout}')

    def generate(self, capture: bool=False, vslite_ctx: dict=None) -> T.Optional[dict]:
        if vslite_ctx:
            raise MesonBugException("We do not expect the ninja backend to be given a valid 'vslite_ctx'")
        ninja = environment.detect_ninja_command_and_version(log=True)
        if self.environment.coredata.get_option(OptionKey('vsenv')):
            builddir = Path(self.environment.get_build_dir())
            try:
                builddir = builddir.relative_to(Path.cwd())
            except ValueError:
                pass
            meson_command = mesonlib.join_args(mesonlib.get_meson_command())
            mlog.log()
            mlog.log('Visual Studio environment is needed to run Ninja. It is recommended to use Meson wrapper:')
            mlog.log(f'{meson_command} compile -C {builddir}')
        if ninja is None:
            raise MesonException('Could not detect Ninja v1.8.2 or newer')
        self.ninja_command, self.ninja_version = ninja
        outfilename = os.path.join(self.environment.get_build_dir(), self.ninja_filename)
        tempfilename = outfilename + '~'
        with open(tempfilename, 'w', encoding='utf-8') as outfile:
            outfile.write(f'# This is the build file for project "{self.build.get_project()}"\n')
            outfile.write('# It is autogenerated by the Meson build system.\n')
            outfile.write('# Do not edit by hand.\n\n')
            outfile.write('ninja_required_version = 1.8.2\n\n')
            num_pools = self.environment.coredata.options[OptionKey('backend_max_links')].value
            if num_pools > 0:
                outfile.write(f'pool link_pool\n  depth = {num_pools}\n\n')
        with self.detect_vs_dep_prefix(tempfilename) as outfile:
            self.generate_rules()
            self.build_elements = []
            self.generate_phony()
            self.add_build_comment(NinjaComment('Build rules for targets'))
            if capture:
                captured_compile_args_per_target = {}
                for target in self.build.get_targets().values():
                    if isinstance(target, build.BuildTarget):
                        captured_compile_args_per_target[target.get_id()] = self.generate_common_compile_args_per_src_type(target)
            for t in ProgressBar(self.build.get_targets().values(), desc='Generating targets'):
                self.generate_target(t)
            mlog.log_timestamp('Targets generated')
            self.add_build_comment(NinjaComment('Test rules'))
            self.generate_tests()
            mlog.log_timestamp('Tests generated')
            self.add_build_comment(NinjaComment('Install rules'))
            self.generate_install()
            mlog.log_timestamp('Install generated')
            self.generate_dist()
            mlog.log_timestamp('Dist generated')
            key = OptionKey('b_coverage')
            if key in self.environment.coredata.options and self.environment.coredata.options[key].value:
                gcovr_exe, gcovr_version, lcov_exe, lcov_version, genhtml_exe, llvm_cov_exe = environment.find_coverage_tools(self.environment.coredata)
                mlog.debug(f'Using {gcovr_exe} ({gcovr_version}), {lcov_exe} and {llvm_cov_exe} for code coverage')
                if gcovr_exe or (lcov_exe and genhtml_exe):
                    self.add_build_comment(NinjaComment('Coverage rules'))
                    self.generate_coverage_rules(gcovr_exe, gcovr_version, llvm_cov_exe)
                    mlog.log_timestamp('Coverage rules generated')
                else:
                    mlog.warning('Need gcovr or lcov/genhtml to generate any coverage reports')
            self.add_build_comment(NinjaComment('Suffix'))
            self.generate_utils()
            mlog.log_timestamp('Utils generated')
            self.generate_ending()
            self.write_rules(outfile)
            self.write_builds(outfile)
            default = 'default all\n\n'
            outfile.write(default)
        os.replace(tempfilename, outfilename)
        mlog.cmd_ci_include(outfilename)
        if mesonlib.version_compare(self.ninja_version, '>=1.10.0') and os.path.exists(os.path.join(self.environment.build_dir, '.ninja_log')):
            subprocess.call(self.ninja_command + ['-t', 'restat'], cwd=self.environment.build_dir)
            subprocess.call(self.ninja_command + ['-t', 'cleandead'], cwd=self.environment.build_dir)
        self.generate_compdb()
        self.generate_rust_project_json()
        if capture:
            return captured_compile_args_per_target

    def generate_rust_project_json(self) -> None:
        """Generate a rust-analyzer compatible rust-project.json file."""
        if not self.rust_crates:
            return
        with open(os.path.join(self.environment.get_build_dir(), 'rust-project.json'), 'w', encoding='utf-8') as f:
            json.dump({'sysroot_src': os.path.join(self.environment.coredata.compilers.host['rust'].get_sysroot(), 'lib/rustlib/src/rust/library/'), 'crates': [c.to_json() for c in self.rust_crates.values()]}, f, indent=4)

    def generate_compdb(self):
        rules = []
        for for_machine in MachineChoice:
            for compiler in self.environment.coredata.compilers[for_machine].values():
                rules += [f'{rule}{ext}' for rule in [self.compiler_to_rule_name(compiler)] for ext in ['', '_RSP']]
                rules += [f'{rule}{ext}' for rule in [self.compiler_to_pch_rule_name(compiler)] for ext in ['', '_RSP']]
        compdb_options = ['-x'] if mesonlib.version_compare(self.ninja_version, '>=1.9') else []
        ninja_compdb = self.ninja_command + ['-t', 'compdb'] + compdb_options + rules
        builddir = self.environment.get_build_dir()
        try:
            jsondb = subprocess.check_output(ninja_compdb, cwd=builddir)
            with open(os.path.join(builddir, 'compile_commands.json'), 'wb') as f:
                f.write(jsondb)
        except Exception:
            mlog.warning('Could not create compilation database.', fatal=False)

    def get_generated_headers(self, target):
        if hasattr(target, 'cached_generated_headers'):
            return target.cached_generated_headers
        header_deps = []
        for genlist in target.get_generated_sources():
            if isinstance(genlist, (build.CustomTarget, build.CustomTargetIndex)):
                continue
            for src in genlist.get_outputs():
                if self.environment.is_header(src):
                    header_deps.append(self.get_target_generated_dir(target, genlist, src))
        if 'vala' in target.compilers and (not isinstance(target, build.Executable)):
            vala_header = File.from_built_file(self.get_target_dir(target), target.vala_header)
            header_deps.append(vala_header)
        for dep in itertools.chain(target.link_targets, target.link_whole_targets):
            if isinstance(dep, (build.StaticLibrary, build.SharedLibrary)):
                header_deps += self.get_generated_headers(dep)
        if isinstance(target, build.CompileTarget):
            header_deps.extend(target.get_generated_headers())
        target.cached_generated_headers = header_deps
        return header_deps

    def get_target_generated_sources(self, target: build.BuildTarget) -> T.MutableMapping[str, File]:
        """
        Returns a dictionary with the keys being the path to the file
        (relative to the build directory) and the value being the File object
        representing the same path.
        """
        srcs: T.MutableMapping[str, File] = OrderedDict()
        for gensrc in target.get_generated_sources():
            for s in gensrc.get_outputs():
                rel_src = self.get_target_generated_dir(target, gensrc, s)
                srcs[rel_src] = File.from_built_relative(rel_src)
        return srcs

    def get_target_sources(self, target: build.BuildTarget) -> T.MutableMapping[str, File]:
        srcs: T.MutableMapping[str, File] = OrderedDict()
        for s in target.get_sources():
            if not isinstance(s, File):
                raise InvalidArguments(f'All sources in target {s!r} must be of type mesonlib.File')
            f = s.rel_to_builddir(self.build_to_src)
            srcs[f] = s
        return srcs

    def get_target_source_can_unity(self, target, source):
        if isinstance(source, File):
            source = source.fname
        if self.environment.is_llvm_ir(source) or self.environment.is_assembly(source):
            return False
        suffix = os.path.splitext(source)[1][1:].lower()
        for lang in backends.LANGS_CANT_UNITY:
            if lang not in target.compilers:
                continue
            if suffix in target.compilers[lang].file_suffixes:
                return False
        return True

    def create_target_source_introspection(self, target: build.Target, comp: compilers.Compiler, parameters, sources, generated_sources, unity_sources: T.Optional[T.List[mesonlib.FileOrString]]=None):
        """
        Adds the source file introspection information for a language of a target

        Internal introspection storage format:
        self.introspection_data = {
            '<target ID>': {
                <id tuple>: {
                    'language: 'lang',
                    'compiler': ['comp', 'exe', 'list'],
                    'parameters': ['UNIQUE', 'parameter', 'list'],
                    'sources': [],
                    'generated_sources': [],
                }
            }
        }
        """
        tid = target.get_id()
        lang = comp.get_language()
        tgt = self.introspection_data[tid]
        id_hash = (lang, tuple(parameters))
        src_block = tgt.get(id_hash, None)
        if src_block is None:
            if isinstance(parameters, CompilerArgs):
                parameters = parameters.to_native(copy=True)
            parameters = comp.compute_parameters_with_absolute_paths(parameters, self.build_dir)
            src_block = {'language': lang, 'compiler': comp.get_exelist(), 'parameters': parameters, 'sources': [], 'generated_sources': [], 'unity_sources': []}
            tgt[id_hash] = src_block

        def compute_path(file: mesonlib.FileOrString) -> str:
            """ Make source files absolute """
            if isinstance(file, File):
                return file.absolute_path(self.source_dir, self.build_dir)
            return os.path.normpath(os.path.join(self.build_dir, file))
        src_block['sources'].extend((compute_path(x) for x in sources))
        src_block['generated_sources'].extend((compute_path(x) for x in generated_sources))
        if unity_sources:
            src_block['unity_sources'].extend((compute_path(x) for x in unity_sources))

    def create_target_linker_introspection(self, target: build.Target, linker: T.Union[Compiler, StaticLinker], parameters):
        tid = target.get_id()
        tgt = self.introspection_data[tid]
        lnk_hash = tuple(parameters)
        lnk_block = tgt.get(lnk_hash, None)
        if lnk_block is None:
            if isinstance(parameters, CompilerArgs):
                parameters = parameters.to_native(copy=True)
            if isinstance(linker, Compiler):
                linkers = linker.get_linker_exelist()
            else:
                linkers = linker.get_exelist()
            lnk_block = {'linker': linkers, 'parameters': parameters}
            tgt[lnk_hash] = lnk_block

    def generate_target(self, target):
        try:
            if isinstance(target, build.BuildTarget):
                os.makedirs(self.get_target_private_dir_abs(target))
        except FileExistsError:
            pass
        if isinstance(target, build.CustomTarget):
            self.generate_custom_target(target)
        if isinstance(target, build.RunTarget):
            self.generate_run_target(target)
        compiled_sources = []
        source2object = {}
        name = target.get_id()
        if name in self.processed_targets:
            return
        self.processed_targets.add(name)
        self.introspection_data[name] = {}
        self.process_target_dependencies(target)
        self.generate_shlib_aliases(target, self.get_target_dir(target))
        if isinstance(target, build.Jar):
            self.generate_jar_target(target)
            return
        if target.uses_rust():
            self.generate_rust_target(target)
            return
        if 'cs' in target.compilers:
            self.generate_cs_target(target)
            return
        if 'swift' in target.compilers:
            self.generate_swift_target(target)
            return
        is_compile_target = isinstance(target, build.CompileTarget)
        target_sources: T.MutableMapping[str, File]
        generated_sources: T.MutableMapping[str, File]
        transpiled_sources: T.List[str]
        if 'vala' in target.compilers:
            target_sources, generated_sources, transpiled_sources = self.generate_vala_compile(target)
        elif 'cython' in target.compilers:
            target_sources, generated_sources, transpiled_sources = self.generate_cython_transpile(target)
        else:
            target_sources = self.get_target_sources(target)
            generated_sources = self.get_target_generated_sources(target)
            transpiled_sources = []
        self.scan_fortran_module_outputs(target)
        self.generate_generator_list_rules(target)
        outname = self.get_target_filename(target)
        obj_list = []
        is_unity = target.is_unity
        header_deps = []
        unity_src = []
        unity_deps = []
        header_deps += self.get_generated_headers(target)
        if is_unity:
            langs = set(target.compilers.keys())
            langs_cant = langs.intersection(backends.LANGS_CANT_UNITY)
            if langs_cant:
                langs_are = langs = ', '.join(langs_cant).upper()
                langs_are += ' are' if len(langs_cant) > 1 else ' is'
                msg = f'{langs_are} not supported in Unity builds yet, so {langs} sources in the {target.name!r} target will be compiled normally'
                mlog.log(mlog.red('FIXME'), msg)
        generated_source_files = []
        for rel_src in generated_sources.keys():
            raw_src = File.from_built_relative(rel_src)
            if self.environment.is_source(rel_src):
                if is_unity and self.get_target_source_can_unity(target, rel_src):
                    unity_deps.append(raw_src)
                    abs_src = os.path.join(self.environment.get_build_dir(), rel_src)
                    unity_src.append(abs_src)
                else:
                    generated_source_files.append(raw_src)
            elif self.environment.is_object(rel_src):
                obj_list.append(rel_src)
            elif self.environment.is_library(rel_src) or modules.is_module_library(rel_src):
                pass
            elif is_compile_target:
                generated_source_files.append(raw_src)
            else:
                header_deps.append(raw_src)
        d_generated_deps = []
        for src in generated_source_files:
            if self.environment.is_llvm_ir(src):
                o, s = self.generate_llvm_ir_compile(target, src)
            else:
                o, s = self.generate_single_compile(target, src, True, order_deps=header_deps)
            compiled_sources.append(s)
            source2object[s] = o
            obj_list.append(o)
            if s.split('.')[-1] in compilers.lang_suffixes['d']:
                d_generated_deps.append(o)
        use_pch = self.target_uses_pch(target)
        if use_pch and target.has_pch():
            pch_objects = self.generate_pch(target, header_deps=header_deps)
        else:
            pch_objects = []
        o, od = self.flatten_object_list(target)
        obj_targets = [t for t in od if t.uses_fortran()]
        obj_list.extend(o)
        fortran_order_deps = [File(True, *os.path.split(self.get_target_filename(t))) for t in obj_targets]
        fortran_inc_args: T.List[str] = []
        if target.uses_fortran():
            fortran_inc_args = mesonlib.listify([target.compilers['fortran'].get_include_args(self.get_target_private_dir(t), is_system=False) for t in obj_targets])
        transpiled_source_files = []
        for src in transpiled_sources:
            raw_src = File.from_built_relative(src)
            if self.environment.is_header(src):
                header_deps.append(raw_src)
            else:
                transpiled_source_files.append(raw_src)
        for src in transpiled_source_files:
            o, s = self.generate_single_compile(target, src, True, [], header_deps)
            obj_list.append(o)
        for src in target_sources.values():
            if not self.environment.is_header(src) or is_compile_target:
                if self.environment.is_llvm_ir(src):
                    o, s = self.generate_llvm_ir_compile(target, src)
                    obj_list.append(o)
                elif is_unity and self.get_target_source_can_unity(target, src):
                    abs_src = os.path.join(self.environment.get_build_dir(), src.rel_to_builddir(self.build_to_src))
                    unity_src.append(abs_src)
                else:
                    o, s = self.generate_single_compile(target, src, False, [], header_deps + d_generated_deps + fortran_order_deps, fortran_inc_args)
                    obj_list.append(o)
                    compiled_sources.append(s)
                    source2object[s] = o
        if is_unity:
            for src in self.generate_unity_files(target, unity_src):
                o, s = self.generate_single_compile(target, src, True, unity_deps + header_deps + d_generated_deps, fortran_order_deps, fortran_inc_args, unity_src)
                obj_list.append(o)
                compiled_sources.append(s)
                source2object[s] = o
        if is_compile_target:
            return
        linker, stdlib_args = self.determine_linker_and_stdlib_args(target)
        if isinstance(target, build.StaticLibrary) and target.prelink:
            final_obj_list = self.generate_prelink(target, obj_list)
        else:
            final_obj_list = obj_list
        elem = self.generate_link(target, outname, final_obj_list, linker, pch_objects, stdlib_args=stdlib_args)
        self.generate_dependency_scan_target(target, compiled_sources, source2object, generated_source_files, fortran_order_deps)
        self.add_build(elem)
        if isinstance(target, build.SharedLibrary) and self.environment.machines[target.for_machine].is_aix():
            if target.aix_so_archive:
                elem = NinjaBuildElement(self.all_outputs, linker.get_archive_name(outname), 'AIX_LINKER', [outname])
                self.add_build(elem)

    def should_use_dyndeps_for_target(self, target: 'build.BuildTarget') -> bool:
        if mesonlib.version_compare(self.ninja_version, '<1.10.0'):
            return False
        if 'fortran' in target.compilers:
            return True
        if 'cpp' not in target.compilers:
            return False
        if '-fmodules-ts' in target.extra_args['cpp']:
            return True
        cpp = target.compilers['cpp']
        if cpp.get_id() != 'msvc':
            return False
        cppversion = target.get_option(OptionKey('std', machine=target.for_machine, lang='cpp'))
        if cppversion not in ('latest', 'c++latest', 'vc++latest'):
            return False
        if not mesonlib.current_vs_supports_modules():
            return False
        if mesonlib.version_compare(cpp.version, '<19.28.28617'):
            return False
        return True

    def generate_dependency_scan_target(self, target: build.BuildTarget, compiled_sources, source2object, generated_source_files: T.List[mesonlib.File], object_deps: T.List['mesonlib.FileOrString']) -> None:
        if not self.should_use_dyndeps_for_target(target):
            return
        depscan_file = self.get_dep_scan_file_for(target)
        pickle_base = target.name + '.dat'
        pickle_file = os.path.join(self.get_target_private_dir(target), pickle_base).replace('\\', '/')
        pickle_abs = os.path.join(self.get_target_private_dir_abs(target), pickle_base).replace('\\', '/')
        json_abs = os.path.join(self.get_target_private_dir_abs(target), f'{target.name}-deps.json').replace('\\', '/')
        rule_name = 'depscan'
        scan_sources = self.select_sources_to_scan(compiled_sources)
        with open(json_abs, 'w', encoding='utf-8') as f:
            json.dump(scan_sources, f)
        elem = NinjaBuildElement(self.all_outputs, depscan_file, rule_name, json_abs)
        elem.add_item('picklefile', pickle_file)
        for g in generated_source_files:
            elem.orderdeps.add(g.relative_name())
        elem.orderdeps.update(object_deps)
        scaninfo = TargetDependencyScannerInfo(self.get_target_private_dir(target), source2object)
        with open(pickle_abs, 'wb') as p:
            pickle.dump(scaninfo, p)
        self.add_build(elem)

    def select_sources_to_scan(self, compiled_sources):
        all_suffixes = set(compilers.lang_suffixes['cpp']) | set(compilers.lang_suffixes['fortran'])
        selected_sources = []
        for source in compiled_sources:
            ext = os.path.splitext(source)[1][1:]
            if ext != 'C':
                ext = ext.lower()
            if ext in all_suffixes:
                selected_sources.append(source)
        return selected_sources

    def process_target_dependencies(self, target):
        for t in target.get_dependencies():
            if t.get_id() not in self.processed_targets:
                self.generate_target(t)

    def custom_target_generator_inputs(self, target):
        for s in target.sources:
            if isinstance(s, build.GeneratedList):
                self.generate_genlist_for_target(s, target)

    def unwrap_dep_list(self, target):
        deps = []
        for i in target.get_dependencies():
            if isinstance(i, list):
                i = i[0]
            for output in i.get_outputs():
                deps.append(os.path.join(self.get_target_dir(i), output))
        return deps

    def generate_custom_target(self, target: build.CustomTarget):
        self.custom_target_generator_inputs(target)
        srcs, ofilenames, cmd = self.eval_custom_target_command(target)
        deps = self.unwrap_dep_list(target)
        deps += self.get_target_depend_files(target)
        if target.build_always_stale:
            deps.append('PHONY')
        if target.depfile is None:
            rulename = 'CUSTOM_COMMAND'
        else:
            rulename = 'CUSTOM_COMMAND_DEP'
        elem = NinjaBuildElement(self.all_outputs, ofilenames, rulename, srcs)
        elem.add_dep(deps)
        for d in target.extra_depends:
            for output in d.get_outputs():
                elem.add_dep(os.path.join(self.get_target_dir(d), output))
        cmd, reason = self.as_meson_exe_cmdline(target.command[0], cmd[1:], extra_bdeps=target.get_transitive_build_target_deps(), capture=ofilenames[0] if target.capture else None, feed=srcs[0] if target.feed else None, env=target.env, verbose=target.console)
        if reason:
            cmd_type = f' (wrapped by meson {reason})'
        else:
            cmd_type = ''
        if target.depfile is not None:
            depfile = target.get_dep_outname(elem.infilenames)
            rel_dfile = os.path.join(self.get_target_dir(target), depfile)
            abs_pdir = os.path.join(self.environment.get_build_dir(), self.get_target_dir(target))
            os.makedirs(abs_pdir, exist_ok=True)
            elem.add_item('DEPFILE', rel_dfile)
        if target.console:
            elem.add_item('pool', 'console')
        full_name = Path(target.subdir, target.name).as_posix()
        elem.add_item('COMMAND', cmd)
        elem.add_item('description', target.description.format(full_name) + cmd_type)
        self.add_build(elem)
        self.processed_targets.add(target.get_id())

    def build_run_target_name(self, target):
        if target.subproject != '':
            subproject_prefix = f'{target.subproject}@@'
        else:
            subproject_prefix = ''
        return f'{subproject_prefix}{target.name}'

    def generate_run_target(self, target: build.RunTarget):
        target_name = self.build_run_target_name(target)
        if not target.command:
            elem = NinjaBuildElement(self.all_outputs, target_name, 'phony', [])
        else:
            target_env = self.get_run_target_env(target)
            _, _, cmd = self.eval_custom_target_command(target)
            meson_exe_cmd, reason = self.as_meson_exe_cmdline(target.command[0], cmd[1:], env=target_env, verbose=True)
            cmd_type = f' (wrapped by meson {reason})' if reason else ''
            elem = self.create_phony_target(target_name, 'CUSTOM_COMMAND', [])
            elem.add_item('COMMAND', meson_exe_cmd)
            elem.add_item('description', f'Running external command {target.name}{cmd_type}')
            elem.add_item('pool', 'console')
        deps = self.unwrap_dep_list(target)
        deps += self.get_target_depend_files(target)
        elem.add_dep(deps)
        self.add_build(elem)
        self.processed_targets.add(target.get_id())

    def generate_coverage_command(self, elem, outputs: T.List[str], gcovr_exe: T.Optional[str], llvm_cov_exe: T.Optional[str]):
        targets = self.build.get_targets().values()
        use_llvm_cov = False
        exe_args = []
        if gcovr_exe is not None:
            exe_args += ['--gcov', gcovr_exe]
        if llvm_cov_exe is not None:
            exe_args += ['--llvm-cov', llvm_cov_exe]
        for target in targets:
            if not hasattr(target, 'compilers'):
                continue
            for compiler in target.compilers.values():
                if compiler.get_id() == 'clang' and (not compiler.info.is_darwin()):
                    use_llvm_cov = True
                    break
        elem.add_item('COMMAND', self.environment.get_build_command() + ['--internal', 'coverage'] + outputs + [self.environment.get_source_dir(), os.path.join(self.environment.get_source_dir(), self.build.get_subproject_dir()), self.environment.get_build_dir(), self.environment.get_log_dir()] + exe_args + (['--use-llvm-cov'] if use_llvm_cov else []))

    def generate_coverage_rules(self, gcovr_exe: T.Optional[str], gcovr_version: T.Optional[str], llvm_cov_exe: T.Optional[str]):
        e = self.create_phony_target('coverage', 'CUSTOM_COMMAND', 'PHONY')
        self.generate_coverage_command(e, [], gcovr_exe, llvm_cov_exe)
        e.add_item('description', 'Generates coverage reports')
        self.add_build(e)
        self.generate_coverage_legacy_rules(gcovr_exe, gcovr_version, llvm_cov_exe)

    def generate_coverage_legacy_rules(self, gcovr_exe: T.Optional[str], gcovr_version: T.Optional[str], llvm_cov_exe: T.Optional[str]):
        e = self.create_phony_target('coverage-html', 'CUSTOM_COMMAND', 'PHONY')
        self.generate_coverage_command(e, ['--html'], gcovr_exe, llvm_cov_exe)
        e.add_item('description', 'Generates HTML coverage report')
        self.add_build(e)
        if gcovr_exe:
            e = self.create_phony_target('coverage-xml', 'CUSTOM_COMMAND', 'PHONY')
            self.generate_coverage_command(e, ['--xml'], gcovr_exe, llvm_cov_exe)
            e.add_item('description', 'Generates XML coverage report')
            self.add_build(e)
            e = self.create_phony_target('coverage-text', 'CUSTOM_COMMAND', 'PHONY')
            self.generate_coverage_command(e, ['--text'], gcovr_exe, llvm_cov_exe)
            e.add_item('description', 'Generates text coverage report')
            self.add_build(e)
            if mesonlib.version_compare(gcovr_version, '>=4.2'):
                e = self.create_phony_target('coverage-sonarqube', 'CUSTOM_COMMAND', 'PHONY')
                self.generate_coverage_command(e, ['--sonarqube'], gcovr_exe, llvm_cov_exe)
                e.add_item('description', 'Generates Sonarqube XML coverage report')
                self.add_build(e)

    def generate_install(self):
        self.create_install_data_files()
        elem = self.create_phony_target('install', 'CUSTOM_COMMAND', 'PHONY')
        elem.add_dep('all')
        elem.add_item('DESC', 'Installing files.')
        elem.add_item('COMMAND', self.environment.get_build_command() + ['install', '--no-rebuild'])
        elem.add_item('pool', 'console')
        self.add_build(elem)

    def generate_tests(self):
        self.serialize_tests()
        cmd = self.environment.get_build_command(True) + ['test', '--no-rebuild']
        if not self.environment.coredata.get_option(OptionKey('stdsplit')):
            cmd += ['--no-stdsplit']
        if self.environment.coredata.get_option(OptionKey('errorlogs')):
            cmd += ['--print-errorlogs']
        elem = self.create_phony_target('test', 'CUSTOM_COMMAND', ['all', 'PHONY'])
        elem.add_item('COMMAND', cmd)
        elem.add_item('DESC', 'Running all tests.')
        elem.add_item('pool', 'console')
        self.add_build(elem)
        cmd = self.environment.get_build_command(True) + ['test', '--benchmark', '--logbase', 'benchmarklog', '--num-processes=1', '--no-rebuild']
        elem = self.create_phony_target('benchmark', 'CUSTOM_COMMAND', ['all', 'PHONY'])
        elem.add_item('COMMAND', cmd)
        elem.add_item('DESC', 'Running benchmark suite.')
        elem.add_item('pool', 'console')
        self.add_build(elem)

    def generate_rules(self):
        self.rules = []
        self.ruledict = {}
        self.add_rule_comment(NinjaComment('Rules for module scanning.'))
        self.generate_scanner_rules()
        self.add_rule_comment(NinjaComment('Rules for compiling.'))
        self.generate_compile_rules()
        self.add_rule_comment(NinjaComment('Rules for linking.'))
        self.generate_static_link_rules()
        self.generate_dynamic_link_rules()
        self.add_rule_comment(NinjaComment('Other rules'))
        self.add_rule(NinjaRule('CUSTOM_COMMAND', ['$COMMAND'], [], '$DESC', extra='restat = 1'))
        self.add_rule(NinjaRule('CUSTOM_COMMAND_DEP', ['$COMMAND'], [], '$DESC', deps='gcc', depfile='$DEPFILE', extra='restat = 1'))
        self.add_rule(NinjaRule('COPY_FILE', self.environment.get_build_command() + ['--internal', 'copy'], ['$in', '$out'], 'Copying $in to $out'))
        c = self.environment.get_build_command() + ['--internal', 'regenerate', self.environment.get_source_dir(), '.']
        self.add_rule(NinjaRule('REGENERATE_BUILD', c, [], 'Regenerating build files.', extra='generator = 1'))

    def add_rule_comment(self, comment: NinjaComment) -> None:
        self.rules.append(comment)

    def add_build_comment(self, comment: NinjaComment) -> None:
        self.build_elements.append(comment)

    def add_rule(self, rule: NinjaRule) -> None:
        if rule.name in self.ruledict:
            raise MesonException(f'Tried to add rule {rule.name} twice.')
        self.rules.append(rule)
        self.ruledict[rule.name] = rule

    def add_build(self, build: NinjaBuildElement) -> None:
        build.check_outputs()
        self.build_elements.append(build)
        if build.rulename != 'phony':
            if build.rulename in self.ruledict:
                build.rule = self.ruledict[build.rulename]
            else:
                mlog.warning(f'build statement for {build.outfilenames} references nonexistent rule {build.rulename}')

    def write_rules(self, outfile: T.TextIO) -> None:
        for b in self.build_elements:
            if isinstance(b, NinjaBuildElement):
                b.count_rule_references()
        for r in self.rules:
            r.write(outfile)

    def write_builds(self, outfile: T.TextIO) -> None:
        for b in ProgressBar(self.build_elements, desc='Writing build.ninja'):
            b.write(outfile)
        mlog.log_timestamp('build.ninja generated')

    def generate_phony(self) -> None:
        self.add_build_comment(NinjaComment('Phony build target, always out of date'))
        elem = NinjaBuildElement(self.all_outputs, 'PHONY', 'phony', '')
        self.add_build(elem)

    def generate_jar_target(self, target: build.Jar):
        fname = target.get_filename()
        outname_rel = os.path.join(self.get_target_dir(target), fname)
        src_list = target.get_sources()
        resources = target.get_java_resources()
        class_list = []
        compiler = target.compilers['java']
        c = 'c'
        m = 'm'
        e = ''
        f = 'f'
        main_class = target.get_main_class()
        if main_class != '':
            e = 'e'
        generated_sources = self.get_target_generated_sources(target)
        gen_src_list = []
        for rel_src in generated_sources.keys():
            raw_src = File.from_built_relative(rel_src)
            if rel_src.endswith('.java'):
                gen_src_list.append(raw_src)
        compile_args = self.determine_single_java_compile_args(target, compiler)
        for src in src_list + gen_src_list:
            plain_class_path = self.generate_single_java_compile(src, target, compiler, compile_args)
            class_list.append(plain_class_path)
        class_dep_list = [os.path.join(self.get_target_private_dir(target), i) for i in class_list]
        manifest_path = os.path.join(self.get_target_private_dir(target), 'META-INF', 'MANIFEST.MF')
        manifest_fullpath = os.path.join(self.environment.get_build_dir(), manifest_path)
        os.makedirs(os.path.dirname(manifest_fullpath), exist_ok=True)
        with open(manifest_fullpath, 'w', encoding='utf-8') as manifest:
            if any(target.link_targets):
                manifest.write('Class-Path: ')
                cp_paths = [os.path.join(self.get_target_dir(l), l.get_filename()) for l in target.link_targets]
                manifest.write(' '.join(cp_paths))
            manifest.write('\n')
        jar_rule = 'java_LINKER'
        commands = [c + m + e + f]
        commands.append(manifest_path)
        if e != '':
            commands.append(main_class)
        commands.append(self.get_target_filename(target))
        commands += ['-C', self.get_target_private_dir(target), '.']
        elem = NinjaBuildElement(self.all_outputs, outname_rel, jar_rule, [])
        elem.add_dep(class_dep_list)
        if resources:
            elem.add_orderdep(self.__generate_sources_structure(Path(self.get_target_private_dir(target)), resources)[0])
        elem.add_item('ARGS', commands)
        self.add_build(elem)
        self.create_target_source_introspection(target, compiler, compile_args, src_list, gen_src_list)

    def generate_cs_resource_tasks(self, target):
        args = []
        deps = []
        for r in target.resources:
            rel_sourcefile = os.path.join(self.build_to_src, target.subdir, r)
            if r.endswith('.resources'):
                a = '-resource:' + rel_sourcefile
            elif r.endswith('.txt') or r.endswith('.resx'):
                ofilebase = os.path.splitext(os.path.basename(r))[0] + '.resources'
                ofilename = os.path.join(self.get_target_private_dir(target), ofilebase)
                elem = NinjaBuildElement(self.all_outputs, ofilename, 'CUSTOM_COMMAND', rel_sourcefile)
                elem.add_item('COMMAND', ['resgen', rel_sourcefile, ofilename])
                elem.add_item('DESC', f'Compiling resource {rel_sourcefile}')
                self.add_build(elem)
                deps.append(ofilename)
                a = '-resource:' + ofilename
            else:
                raise InvalidArguments(f'Unknown resource file {r}.')
            args.append(a)
        return (args, deps)

    def generate_cs_target(self, target: build.BuildTarget):
        fname = target.get_filename()
        outname_rel = os.path.join(self.get_target_dir(target), fname)
        src_list = target.get_sources()
        compiler = target.compilers['cs']
        rel_srcs = [os.path.normpath(s.rel_to_builddir(self.build_to_src)) for s in src_list]
        deps = []
        commands = compiler.compiler_args(target.extra_args['cs'])
        commands += compiler.get_optimization_args(target.get_option(OptionKey('optimization')))
        commands += compiler.get_debug_args(target.get_option(OptionKey('debug')))
        if isinstance(target, build.Executable):
            commands.append('-target:exe')
        elif isinstance(target, build.SharedLibrary):
            commands.append('-target:library')
        else:
            raise MesonException('Unknown C# target type.')
        resource_args, resource_deps = self.generate_cs_resource_tasks(target)
        commands += resource_args
        deps += resource_deps
        commands += compiler.get_output_args(outname_rel)
        for l in target.link_targets:
            lname = os.path.join(self.get_target_dir(l), l.get_filename())
            commands += compiler.get_link_args(lname)
            deps.append(lname)
        if '-g' in commands:
            outputs = [outname_rel, outname_rel + '.mdb']
        else:
            outputs = [outname_rel]
        generated_sources = self.get_target_generated_sources(target)
        generated_rel_srcs = []
        for rel_src in generated_sources.keys():
            if rel_src.lower().endswith('.cs'):
                generated_rel_srcs.append(os.path.normpath(rel_src))
            deps.append(os.path.normpath(rel_src))
        for dep in target.get_external_deps():
            commands.extend_direct(dep.get_link_args())
        commands += self.build.get_project_args(compiler, target.subproject, target.for_machine)
        commands += self.build.get_global_args(compiler, target.for_machine)
        elem = NinjaBuildElement(self.all_outputs, outputs, self.compiler_to_rule_name(compiler), rel_srcs + generated_rel_srcs)
        elem.add_dep(deps)
        elem.add_item('ARGS', commands)
        self.add_build(elem)
        self.generate_generator_list_rules(target)
        self.create_target_source_introspection(target, compiler, commands, rel_srcs, generated_rel_srcs)

    def determine_single_java_compile_args(self, target, compiler):
        args = []
        args += self.build.get_global_args(compiler, target.for_machine)
        args += self.build.get_project_args(compiler, target.subproject, target.for_machine)
        args += target.get_java_args()
        args += compiler.get_output_args(self.get_target_private_dir(target))
        args += target.get_classpath_args()
        curdir = target.get_subdir()
        sourcepath = os.path.join(self.build_to_src, curdir) + os.pathsep
        sourcepath += os.path.normpath(curdir) + os.pathsep
        for i in target.include_dirs:
            for idir in i.get_incdirs():
                sourcepath += os.path.join(self.build_to_src, i.curdir, idir) + os.pathsep
        args += ['-sourcepath', sourcepath]
        return args

    def generate_single_java_compile(self, src, target, compiler, args):
        deps = [os.path.join(self.get_target_dir(l), l.get_filename()) for l in target.link_targets]
        generated_sources = self.get_target_generated_sources(target)
        for rel_src in generated_sources.keys():
            if rel_src.endswith('.java'):
                deps.append(rel_src)
        rel_src = src.rel_to_builddir(self.build_to_src)
        plain_class_path = src.fname[:-4] + 'class'
        rel_obj = os.path.join(self.get_target_private_dir(target), plain_class_path)
        element = NinjaBuildElement(self.all_outputs, rel_obj, self.compiler_to_rule_name(compiler), rel_src)
        element.add_dep(deps)
        element.add_item('ARGS', args)
        self.add_build(element)
        return plain_class_path

    def generate_java_link(self):
        rule = 'java_LINKER'
        command = ['jar', '$ARGS']
        description = 'Creating JAR $out'
        self.add_rule(NinjaRule(rule, command, [], description))

    def determine_dep_vapis(self, target):
        """
        Peek into the sources of BuildTargets we're linking with, and if any of
        them was built with Vala, assume that it also generated a .vapi file of
        the same name as the BuildTarget and return the path to it relative to
        the build directory.
        """
        result = OrderedSet()
        for dep in itertools.chain(target.link_targets, target.link_whole_targets):
            if not dep.is_linkable_target():
                continue
            for i in dep.sources:
                if hasattr(i, 'fname'):
                    i = i.fname
                if i.split('.')[-1] in compilers.lang_suffixes['vala']:
                    vapiname = dep.vala_vapi
                    fullname = os.path.join(self.get_target_dir(dep), vapiname)
                    result.add(fullname)
                    break
        return list(result)

    def split_vala_sources(self, t: build.BuildTarget) -> T.Tuple[T.MutableMapping[str, File], T.MutableMapping[str, File], T.Tuple[T.MutableMapping[str, File], T.MutableMapping]]:
        """
        Splits the target's sources into .vala, .gs, .vapi, and other sources.
        Handles both preexisting and generated sources.

        Returns a tuple (vala, vapi, others) each of which is a dictionary with
        the keys being the path to the file (relative to the build directory)
        and the value being the object that generated or represents the file.
        """
        vala: T.MutableMapping[str, File] = OrderedDict()
        vapi: T.MutableMapping[str, File] = OrderedDict()
        others: T.MutableMapping[str, File] = OrderedDict()
        othersgen: T.MutableMapping[str, File] = OrderedDict()
        for s in t.get_sources():
            if not isinstance(s, File):
                raise InvalidArguments(f'All sources in target {t!r} must be of type mesonlib.File, not {s!r}')
            f = s.rel_to_builddir(self.build_to_src)
            if s.endswith(('.vala', '.gs')):
                srctype = vala
            elif s.endswith('.vapi'):
                srctype = vapi
            else:
                srctype = others
            srctype[f] = s
        for gensrc in t.get_generated_sources():
            for s in gensrc.get_outputs():
                f = self.get_target_generated_dir(t, gensrc, s)
                if s.endswith(('.vala', '.gs')):
                    srctype = vala
                elif s.endswith('.vapi'):
                    srctype = vapi
                else:
                    srctype = othersgen
                if f in srctype and srctype[f] is not gensrc:
                    msg = 'Duplicate output {0!r} from {1!r} {2!r}; conflicts with {0!r} from {4!r} {3!r}'.format(f, type(gensrc).__name__, gensrc.name, srctype[f].name, type(srctype[f]).__name__)
                    raise InvalidArguments(msg)
                srctype[f] = gensrc
        return (vala, vapi, (others, othersgen))

    def generate_vala_compile(self, target: build.BuildTarget) -> T.Tuple[T.MutableMapping[str, File], T.MutableMapping[str, File], T.List[str]]:
        """Vala is compiled into C. Set up all necessary build steps here."""
        vala_src, vapi_src, other_src = self.split_vala_sources(target)
        extra_dep_files = []
        if not vala_src:
            raise InvalidArguments(f'Vala library {target.name!r} has no Vala or Genie source files.')
        valac = target.compilers['vala']
        c_out_dir = self.get_target_private_dir(target)
        vala_c_src: T.List[str] = []
        valac_outputs: T.List = []
        all_files = list(vapi_src)
        srcbasedir = os.path.join(self.build_to_src, target.get_subdir())
        for vala_file, gensrc in vala_src.items():
            all_files.append(vala_file)
            if isinstance(gensrc, (build.CustomTarget, build.GeneratedList)) or gensrc.is_built:
                vala_c_file = os.path.splitext(os.path.basename(vala_file))[0] + '.c'
                abs_srcbasedir = os.path.join(self.environment.get_source_dir(), target.get_subdir())
                abs_vala_file = os.path.join(self.environment.get_build_dir(), vala_file)
                if PurePath(os.path.commonpath((abs_srcbasedir, abs_vala_file))) == PurePath(abs_srcbasedir):
                    vala_c_subdir = PurePath(abs_vala_file).parent.relative_to(abs_srcbasedir)
                    vala_c_file = os.path.join(str(vala_c_subdir), vala_c_file)
            else:
                path_to_target = os.path.join(self.build_to_src, target.get_subdir())
                if vala_file.startswith(path_to_target):
                    vala_c_file = os.path.splitext(os.path.relpath(vala_file, path_to_target))[0] + '.c'
                else:
                    vala_c_file = os.path.splitext(os.path.basename(vala_file))[0] + '.c'
            vala_c_file = os.path.join(c_out_dir, vala_c_file)
            vala_c_src.append(vala_c_file)
            valac_outputs.append(vala_c_file)
        args = self.generate_basic_compiler_args(target, valac)
        args += valac.get_colorout_args(target.get_option(OptionKey('b_colorout')))
        args += ['--directory', c_out_dir]
        args += ['--basedir', srcbasedir]
        if target.is_linkable_target():
            args += ['--library', target.name]
            hname = os.path.join(self.get_target_dir(target), target.vala_header)
            args += ['--header', hname]
            if target.is_unity:
                args += ['--use-header']
            valac_outputs.append(hname)
            vapiname = os.path.join(self.get_target_dir(target), target.vala_vapi)
            args += ['--vapi', os.path.join('..', target.vala_vapi)]
            valac_outputs.append(vapiname)
            if len(target.install_dir) > 1 and target.install_dir[1] is True:
                target.install_dir[1] = self.environment.get_includedir()
            if len(target.install_dir) > 2 and target.install_dir[2] is True:
                target.install_dir[2] = os.path.join(self.environment.get_datadir(), 'vala', 'vapi')
            if isinstance(target.vala_gir, str):
                girname = os.path.join(self.get_target_dir(target), target.vala_gir)
                args += ['--gir', os.path.join('..', target.vala_gir)]
                valac_outputs.append(girname)
                if len(target.install_dir) > 3 and target.install_dir[3] is True:
                    target.install_dir[3] = os.path.join(self.environment.get_datadir(), 'gir-1.0')
        gres_dirs = []
        for gensrc in other_src[1].values():
            if isinstance(gensrc, modules.GResourceTarget):
                gres_xml, = self.get_custom_target_sources(gensrc)
                args += ['--gresources=' + gres_xml]
                for source_dir in gensrc.source_dirs:
                    gres_dirs += [os.path.join(self.get_target_dir(gensrc), source_dir)]
                gres_c, = gensrc.get_outputs()
                extra_dep_files += [os.path.join(self.get_target_dir(gensrc), gres_c)]
        for gres_dir in OrderedSet(gres_dirs):
            args += [f'--gresourcesdir={gres_dir}']
        dependency_vapis = self.determine_dep_vapis(target)
        extra_dep_files += dependency_vapis
        extra_dep_files.extend(self.get_target_depend_files(target))
        args += target.get_extra_args('vala')
        element = NinjaBuildElement(self.all_outputs, valac_outputs, self.compiler_to_rule_name(valac), all_files + dependency_vapis)
        element.add_item('ARGS', args)
        element.add_dep(extra_dep_files)
        self.add_build(element)
        self.create_target_source_introspection(target, valac, args, all_files, [])
        return (other_src[0], other_src[1], vala_c_src)

    def generate_cython_transpile(self, target: build.BuildTarget) -> T.Tuple[T.MutableMapping[str, File], T.MutableMapping[str, File], T.List[str]]:
        """Generate rules for transpiling Cython files to C or C++"""
        static_sources: T.MutableMapping[str, File] = OrderedDict()
        generated_sources: T.MutableMapping[str, File] = OrderedDict()
        cython_sources: T.List[str] = []
        cython = target.compilers['cython']
        args: T.List[str] = []
        args += cython.get_always_args()
        args += cython.get_debug_args(target.get_option(OptionKey('debug')))
        args += cython.get_optimization_args(target.get_option(OptionKey('optimization')))
        args += cython.get_option_compile_args(target.get_options())
        args += self.build.get_global_args(cython, target.for_machine)
        args += self.build.get_project_args(cython, target.subproject, target.for_machine)
        args += target.get_extra_args('cython')
        ext = target.get_option(OptionKey('language', machine=target.for_machine, lang='cython'))
        pyx_sources = []
        for src in target.get_sources():
            if src.endswith('.pyx'):
                output = os.path.join(self.get_target_private_dir(target), f'{src}.{ext}')
                element = NinjaBuildElement(self.all_outputs, [output], self.compiler_to_rule_name(cython), [src.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())])
                element.add_item('ARGS', args)
                self.add_build(element)
                cython_sources.append(output)
                pyx_sources.append(element)
            else:
                static_sources[src.rel_to_builddir(self.build_to_src)] = src
        header_deps = []
        for gen in target.get_generated_sources():
            for ssrc in gen.get_outputs():
                if isinstance(gen, GeneratedList):
                    ssrc = os.path.join(self.get_target_private_dir(target), ssrc)
                else:
                    ssrc = os.path.join(gen.get_subdir(), ssrc)
                if ssrc.endswith('.pyx'):
                    output = os.path.join(self.get_target_private_dir(target), f'{ssrc}.{ext}')
                    element = NinjaBuildElement(self.all_outputs, [output], self.compiler_to_rule_name(cython), [ssrc])
                    element.add_item('ARGS', args)
                    self.add_build(element)
                    pyx_sources.append(element)
                    cython_sources.append(output)
                else:
                    generated_sources[ssrc] = mesonlib.File.from_built_file(gen.get_subdir(), ssrc)
                    if not self.environment.is_source(ssrc) and (not self.environment.is_object(ssrc)) and (not self.environment.is_library(ssrc)) and (not modules.is_module_library(ssrc)):
                        header_deps.append(ssrc)
        for source in pyx_sources:
            source.add_orderdep(header_deps)
        return (static_sources, generated_sources, cython_sources)

    def _generate_copy_target(self, src: 'mesonlib.FileOrString', output: Path) -> None:
        """Create a target to copy a source file from one location to another."""
        if isinstance(src, File):
            instr = src.absolute_path(self.environment.source_dir, self.environment.build_dir)
        else:
            instr = src
        elem = NinjaBuildElement(self.all_outputs, [str(output)], 'COPY_FILE', [instr])
        elem.add_orderdep(instr)
        self.add_build(elem)

    def __generate_sources_structure(self, root: Path, structured_sources: build.StructuredSources) -> T.Tuple[T.List[str], T.Optional[str]]:
        first_file: T.Optional[str] = None
        orderdeps: T.List[str] = []
        for path, files in structured_sources.sources.items():
            for file in files:
                if isinstance(file, File):
                    out = root / path / Path(file.fname).name
                    orderdeps.append(str(out))
                    self._generate_copy_target(file, out)
                    if first_file is None:
                        first_file = str(out)
                else:
                    for f in file.get_outputs():
                        out = root / path / f
                        orderdeps.append(str(out))
                        self._generate_copy_target(str(Path(file.subdir) / f), out)
                        if first_file is None:
                            first_file = str(out)
        return (orderdeps, first_file)

    def _add_rust_project_entry(self, name: str, main_rust_file: str, args: CompilerArgs, from_subproject: bool, proc_macro_dylib_path: T.Optional[str], deps: T.List[RustDep]) -> None:
        raw_edition: T.Optional[str] = mesonlib.first(reversed(args), lambda x: x.startswith('--edition'))
        edition: RUST_EDITIONS = '2015' if not raw_edition else raw_edition.split('=')[-1]
        cfg: T.List[str] = []
        arg_itr: T.Iterator[str] = iter(args)
        for arg in arg_itr:
            if arg == '--cfg':
                cfg.append(next(arg_itr))
            elif arg.startswith('--cfg'):
                cfg.append(arg[len('--cfg'):])
        crate = RustCrate(len(self.rust_crates), name, main_rust_file, edition, deps, cfg, is_workspace_member=not from_subproject, is_proc_macro=proc_macro_dylib_path is not None, proc_macro_dylib_path=proc_macro_dylib_path)
        self.rust_crates[name] = crate

    def _get_rust_dependency_name(self, target: build.BuildTarget, dependency: LibTypes) -> str:
        return target.rust_dependency_map.get(dependency.name, dependency.name).replace('-', '_')

    def generate_rust_target(self, target: build.BuildTarget) -> None:
        rustc = target.compilers['rust']
        base_proxy = target.get_options()
        args = rustc.compiler_args()
        args += compilers.get_base_compile_args(base_proxy, rustc)
        self.generate_generator_list_rules(target)
        deps: T.List[str] = []
        project_deps: T.List[RustDep] = []
        orderdeps: T.List[str] = []
        main_rust_file = None
        if target.structured_sources:
            if target.structured_sources.needs_copy():
                _ods, main_rust_file = self.__generate_sources_structure(Path(self.get_target_private_dir(target)) / 'structured', target.structured_sources)
                orderdeps.extend(_ods)
            else:
                g = target.structured_sources.first_file()
                if isinstance(g, File):
                    main_rust_file = g.rel_to_builddir(self.build_to_src)
                elif isinstance(g, GeneratedList):
                    main_rust_file = os.path.join(self.get_target_private_dir(target), g.get_outputs()[0])
                else:
                    main_rust_file = os.path.join(g.get_subdir(), g.get_outputs()[0])
                for f in target.structured_sources.as_list():
                    if isinstance(f, File):
                        orderdeps.append(f.rel_to_builddir(self.build_to_src))
                    else:
                        orderdeps.extend([os.path.join(self.build_to_src, f.subdir, s) for s in f.get_outputs()])
        for i in target.get_sources():
            if not rustc.can_compile(i):
                raise InvalidArguments(f'Rust target {target.get_basename()} contains a non-rust source file.')
            if main_rust_file is None:
                main_rust_file = i.rel_to_builddir(self.build_to_src)
        for g in target.get_generated_sources():
            for i in g.get_outputs():
                if not rustc.can_compile(i):
                    raise InvalidArguments(f'Rust target {target.get_basename()} contains a non-rust source file.')
                if isinstance(g, GeneratedList):
                    fname = os.path.join(self.get_target_private_dir(target), i)
                else:
                    fname = os.path.join(g.get_subdir(), i)
                if main_rust_file is None:
                    main_rust_file = fname
                orderdeps.append(fname)
        if main_rust_file is None:
            raise RuntimeError('A Rust target has no Rust sources. This is weird. Also a bug. Please report')
        target_name = os.path.join(target.subdir, target.get_filename())
        cratetype = target.rust_crate_type
        args.extend(['--crate-type', cratetype])
        if cratetype in {'bin', 'dylib'}:
            args.extend(rustc.get_linker_always_args())
        args += self.generate_basic_compiler_args(target, rustc)
        args += ['--crate-name', target.name.replace('-', '_').replace(' ', '_').replace('.', '_')]
        depfile = os.path.join(target.subdir, target.name + '.d')
        args += ['--emit', f'dep-info={depfile}', '--emit', f'link={target_name}']
        args += ['--out-dir', self.get_target_private_dir(target)]
        args += ['-C', 'metadata=' + target.get_id()]
        args += target.get_extra_args('rust')
        if not isinstance(target, build.StaticLibrary):
            try:
                buildtype = target.get_option(OptionKey('buildtype'))
                crt = target.get_option(OptionKey('b_vscrt'))
                args += rustc.get_crt_link_args(crt, buildtype)
            except KeyError:
                pass
        if mesonlib.version_compare(rustc.version, '>= 1.67.0'):
            verbatim = '+verbatim'
        else:
            verbatim = ''

        def _link_library(libname: str, static: bool, bundle: bool=False):
            type_ = 'static' if static else 'dylib'
            modifiers = []
            if not bundle and static:
                modifiers.append('-bundle')
            if verbatim:
                modifiers.append(verbatim)
            if modifiers:
                type_ += ':' + ','.join(modifiers)
            args.append(f'-l{type_}={libname}')
        linkdirs = mesonlib.OrderedSet()
        external_deps = target.external_deps.copy()
        target_deps = target.get_dependencies()
        for d in target_deps:
            linkdirs.add(d.subdir)
            deps.append(self.get_dependency_filename(d))
            if isinstance(d, build.StaticLibrary):
                external_deps.extend(d.external_deps)
            if d.uses_rust_abi():
                if d not in itertools.chain(target.link_targets, target.link_whole_targets):
                    continue
                d_name = self._get_rust_dependency_name(target, d)
                args += ['--extern', '{}={}'.format(d_name, os.path.join(d.subdir, d.filename))]
                project_deps.append(RustDep(d_name, self.rust_crates[d.name].order))
                continue
            lib = self.get_target_filename_for_linking(d)
            link_whole = d in target.link_whole_targets
            if isinstance(target, build.StaticLibrary) or (isinstance(target, build.Executable) and rustc.get_crt_static()):
                static = isinstance(d, build.StaticLibrary)
                libname = os.path.basename(lib) if verbatim else d.name
                _link_library(libname, static, bundle=link_whole)
            elif link_whole:
                link_whole_args = rustc.linker.get_link_whole_for([lib])
                args += [f'-Clink-arg={a}' for a in link_whole_args]
            else:
                args.append(f'-Clink-arg={lib}')
        for e in external_deps:
            for a in e.get_link_args():
                if a in rustc.native_static_libs:
                    pass
                elif a.startswith('-L'):
                    args.append(a)
                elif a.endswith(('.dll', '.so', '.dylib', '.a', '.lib')) and isinstance(target, build.StaticLibrary):
                    dir_, lib = os.path.split(a)
                    linkdirs.add(dir_)
                    if not verbatim:
                        lib, ext = os.path.splitext(lib)
                        if lib.startswith('lib'):
                            lib = lib[3:]
                    static = a.endswith(('.a', '.lib'))
                    _link_library(lib, static)
                else:
                    args.append(f'-Clink-arg={a}')
        for d in linkdirs:
            d = d or '.'
            args.append(f'-L{d}')
        args.extend((f'-Clink-arg={a}' for a in target.get_used_stdlib_args('rust')))
        has_shared_deps = any((isinstance(dep, build.SharedLibrary) for dep in target_deps))
        has_rust_shared_deps = any((dep.uses_rust() and dep.rust_crate_type == 'dylib' for dep in target_deps))
        if cratetype in {'dylib', 'proc-macro'} or has_rust_shared_deps:
            args += ['-C', 'prefer-dynamic']
        if isinstance(target, build.SharedLibrary) or has_shared_deps:
            if has_path_sep(target.name):
                target_slashname_workaround_dir = os.path.join(os.path.dirname(target.name), self.get_target_dir(target))
            else:
                target_slashname_workaround_dir = self.get_target_dir(target)
            rpath_args, target.rpath_dirs_to_remove = rustc.build_rpath_args(self.environment, self.environment.get_build_dir(), target_slashname_workaround_dir, self.determine_rpath_dirs(target), target.build_rpath, target.install_rpath)
            for rpath_arg in rpath_args:
                args += ['-C', 'link-arg=' + rpath_arg + ':' + os.path.join(rustc.get_sysroot(), 'lib')]
        proc_macro_dylib_path = None
        if getattr(target, 'rust_crate_type', '') == 'proc-macro':
            proc_macro_dylib_path = os.path.abspath(os.path.join(target.subdir, target.get_filename()))
        self._add_rust_project_entry(target.name, os.path.abspath(os.path.join(self.environment.build_dir, main_rust_file)), args, bool(target.subproject), proc_macro_dylib_path, project_deps)
        compiler_name = self.compiler_to_rule_name(rustc)
        element = NinjaBuildElement(self.all_outputs, target_name, compiler_name, main_rust_file)
        if orderdeps:
            element.add_orderdep(orderdeps)
        if deps:
            element.add_dep(deps)
        element.add_item('ARGS', args)
        element.add_item('targetdep', depfile)
        element.add_item('cratetype', cratetype)
        self.add_build(element)
        if isinstance(target, build.SharedLibrary):
            self.generate_shsym(target)
        self.create_target_source_introspection(target, rustc, args, [main_rust_file], [])

    @staticmethod
    def get_rule_suffix(for_machine: MachineChoice) -> str:
        return PerMachine('_FOR_BUILD', '')[for_machine]

    @classmethod
    def get_compiler_rule_name(cls, lang: str, for_machine: MachineChoice, mode: str='COMPILER') -> str:
        return f'{lang}_{mode}{cls.get_rule_suffix(for_machine)}'

    @classmethod
    def compiler_to_rule_name(cls, compiler: Compiler) -> str:
        return cls.get_compiler_rule_name(compiler.get_language(), compiler.for_machine, compiler.mode)

    @classmethod
    def compiler_to_pch_rule_name(cls, compiler: Compiler) -> str:
        return cls.get_compiler_rule_name(compiler.get_language(), compiler.for_machine, 'PCH')

    def swift_module_file_name(self, target):
        return os.path.join(self.get_target_private_dir(target), self.target_swift_modulename(target) + '.swiftmodule')

    def target_swift_modulename(self, target):
        return target.name

    def determine_swift_dep_modules(self, target):
        result = []
        for l in target.link_targets:
            if self.is_swift_target(l):
                result.append(self.swift_module_file_name(l))
        return result

    def get_swift_link_deps(self, target):
        result = []
        for l in target.link_targets:
            result.append(self.get_target_filename(l))
        return result

    def split_swift_generated_sources(self, target):
        all_srcs = self.get_target_generated_sources(target)
        srcs = []
        others = []
        for i in all_srcs:
            if i.endswith('.swift'):
                srcs.append(i)
            else:
                others.append(i)
        return (srcs, others)

    def generate_swift_target(self, target):
        module_name = self.target_swift_modulename(target)
        swiftc = target.compilers['swift']
        abssrc = []
        relsrc = []
        abs_headers = []
        header_imports = []
        for i in target.get_sources():
            if swiftc.can_compile(i):
                rels = i.rel_to_builddir(self.build_to_src)
                abss = os.path.normpath(os.path.join(self.environment.get_build_dir(), rels))
                relsrc.append(rels)
                abssrc.append(abss)
            elif self.environment.is_header(i):
                relh = i.rel_to_builddir(self.build_to_src)
                absh = os.path.normpath(os.path.join(self.environment.get_build_dir(), relh))
                abs_headers.append(absh)
                header_imports += swiftc.get_header_import_args(absh)
            else:
                raise InvalidArguments(f'Swift target {target.get_basename()} contains a non-swift source file.')
        os.makedirs(self.get_target_private_dir_abs(target), exist_ok=True)
        compile_args = swiftc.get_compile_only_args()
        compile_args += swiftc.get_optimization_args(target.get_option(OptionKey('optimization')))
        compile_args += swiftc.get_debug_args(target.get_option(OptionKey('debug')))
        compile_args += swiftc.get_module_args(module_name)
        compile_args += self.build.get_project_args(swiftc, target.subproject, target.for_machine)
        compile_args += self.build.get_global_args(swiftc, target.for_machine)
        for i in reversed(target.get_include_dirs()):
            basedir = i.get_curdir()
            for d in i.get_incdirs():
                if d not in ('', '.'):
                    expdir = os.path.join(basedir, d)
                else:
                    expdir = basedir
                srctreedir = os.path.normpath(os.path.join(self.environment.get_build_dir(), self.build_to_src, expdir))
                sargs = swiftc.get_include_args(srctreedir, False)
                compile_args += sargs
        compile_args += target.get_extra_args('swift')
        link_args = swiftc.get_output_args(os.path.join(self.environment.get_build_dir(), self.get_target_filename(target)))
        link_args += self.build.get_project_link_args(swiftc, target.subproject, target.for_machine)
        link_args += self.build.get_global_link_args(swiftc, target.for_machine)
        rundir = self.get_target_private_dir(target)
        out_module_name = self.swift_module_file_name(target)
        in_module_files = self.determine_swift_dep_modules(target)
        abs_module_dirs = self.determine_swift_dep_dirs(target)
        module_includes = []
        for x in abs_module_dirs:
            module_includes += swiftc.get_include_args(x, False)
        link_deps = self.get_swift_link_deps(target)
        abs_link_deps = [os.path.join(self.environment.get_build_dir(), x) for x in link_deps]
        for d in target.link_targets:
            reldir = self.get_target_dir(d)
            if reldir == '':
                reldir = '.'
            link_args += ['-L', os.path.normpath(os.path.join(self.environment.get_build_dir(), reldir))]
        rel_generated, _ = self.split_swift_generated_sources(target)
        abs_generated = [os.path.join(self.environment.get_build_dir(), x) for x in rel_generated]
        objects = []
        rel_objects = []
        for i in abssrc + abs_generated:
            base = os.path.basename(i)
            oname = os.path.splitext(base)[0] + '.o'
            objects.append(oname)
            rel_objects.append(os.path.join(self.get_target_private_dir(target), oname))
        rulename = self.compiler_to_rule_name(swiftc)
        elem = NinjaBuildElement(self.all_outputs, rel_objects, rulename, abssrc)
        elem.add_dep(in_module_files + rel_generated)
        elem.add_dep(abs_headers)
        elem.add_item('ARGS', compile_args + header_imports + abs_generated + module_includes)
        elem.add_item('RUNDIR', rundir)
        self.add_build(elem)
        elem = NinjaBuildElement(self.all_outputs, out_module_name, rulename, abssrc)
        elem.add_dep(in_module_files + rel_generated)
        elem.add_item('ARGS', compile_args + abs_generated + module_includes + swiftc.get_mod_gen_args())
        elem.add_item('RUNDIR', rundir)
        self.add_build(elem)
        if isinstance(target, build.StaticLibrary):
            elem = self.generate_link(target, self.get_target_filename(target), rel_objects, self.build.static_linker[target.for_machine])
            self.add_build(elem)
        elif isinstance(target, build.Executable):
            elem = NinjaBuildElement(self.all_outputs, self.get_target_filename(target), rulename, [])
            elem.add_dep(rel_objects)
            elem.add_dep(link_deps)
            elem.add_item('ARGS', link_args + swiftc.get_std_exe_link_args() + objects + abs_link_deps)
            elem.add_item('RUNDIR', rundir)
            self.add_build(elem)
        else:
            raise MesonException('Swift supports only executable and static library targets.')
        self.create_target_source_introspection(target, swiftc, compile_args + header_imports + module_includes, relsrc, rel_generated)

    def _rsp_options(self, tool: T.Union['Compiler', 'StaticLinker', 'DynamicLinker']) -> T.Dict[str, T.Union[bool, RSPFileSyntax]]:
        """Helper method to get rsp options.

        rsp_file_syntax() is only guaranteed to be implemented if
        can_linker_accept_rsp() returns True.
        """
        options = {'rspable': tool.can_linker_accept_rsp()}
        if options['rspable']:
            options['rspfile_quote_style'] = tool.rsp_file_syntax()
        return options

    def generate_static_link_rules(self):
        num_pools = self.environment.coredata.options[OptionKey('backend_max_links')].value
        if 'java' in self.environment.coredata.compilers.host:
            self.generate_java_link()
        for for_machine in MachineChoice:
            static_linker = self.build.static_linker[for_machine]
            if static_linker is None:
                continue
            rule = 'STATIC_LINKER{}'.format(self.get_rule_suffix(for_machine))
            cmdlist: T.List[T.Union[str, NinjaCommandArg]] = []
            args = ['$in']
            if isinstance(static_linker, ArLikeLinker) and (not mesonlib.is_windows()):
                cmdlist = execute_wrapper + [c.format('$out') for c in rmfile_prefix]
            cmdlist += static_linker.get_exelist()
            cmdlist += ['$LINK_ARGS']
            cmdlist += NinjaCommandArg.list(static_linker.get_output_args('$out'), Quoting.none)
            if static_linker.id == 'applear':
                cmdlist.extend(args)
                args = []
                ranlib = self.environment.lookup_binary_entry(for_machine, 'ranlib')
                if ranlib is None:
                    ranlib = ['ranlib']
                cmdlist.extend(['&&'] + ranlib + ['-c', '$out'])
            description = 'Linking static target $out'
            if num_pools > 0:
                pool = 'pool = link_pool'
            else:
                pool = None
            options = self._rsp_options(static_linker)
            self.add_rule(NinjaRule(rule, cmdlist, args, description, **options, extra=pool))

    def generate_dynamic_link_rules(self):
        num_pools = self.environment.coredata.options[OptionKey('backend_max_links')].value
        for for_machine in MachineChoice:
            complist = self.environment.coredata.compilers[for_machine]
            for langname, compiler in complist.items():
                if langname in {'java', 'vala', 'rust', 'cs', 'cython'}:
                    continue
                rule = '{}_LINKER{}'.format(langname, self.get_rule_suffix(for_machine))
                command = compiler.get_linker_exelist()
                args = ['$ARGS'] + NinjaCommandArg.list(compiler.get_linker_output_args('$out'), Quoting.none) + ['$in', '$LINK_ARGS']
                description = 'Linking target $out'
                if num_pools > 0:
                    pool = 'pool = link_pool'
                else:
                    pool = None
                options = self._rsp_options(compiler)
                self.add_rule(NinjaRule(rule, command, args, description, **options, extra=pool))
            if self.environment.machines[for_machine].is_aix():
                rule = 'AIX_LINKER{}'.format(self.get_rule_suffix(for_machine))
                description = 'Archiving AIX shared library'
                cmdlist = compiler.get_command_to_archive_shlib()
                args = []
                options = {}
                self.add_rule(NinjaRule(rule, cmdlist, args, description, **options, extra=None))
        args = self.environment.get_build_command() + ['--internal', 'symbolextractor', self.environment.get_build_dir(), '$in', '$IMPLIB', '$out']
        symrule = 'SHSYM'
        symcmd = args + ['$CROSS']
        syndesc = 'Generating symbol file $out'
        synstat = 'restat = 1'
        self.add_rule(NinjaRule(symrule, symcmd, [], syndesc, extra=synstat))

    def generate_java_compile_rule(self, compiler):
        rule = self.compiler_to_rule_name(compiler)
        command = compiler.get_exelist() + ['$ARGS', '$in']
        description = 'Compiling Java object $in'
        self.add_rule(NinjaRule(rule, command, [], description))

    def generate_cs_compile_rule(self, compiler: 'CsCompiler') -> None:
        rule = self.compiler_to_rule_name(compiler)
        command = compiler.get_exelist()
        args = ['$ARGS', '$in']
        description = 'Compiling C Sharp target $out'
        self.add_rule(NinjaRule(rule, command, args, description, rspable=mesonlib.is_windows(), rspfile_quote_style=compiler.rsp_file_syntax()))

    def generate_vala_compile_rules(self, compiler):
        rule = self.compiler_to_rule_name(compiler)
        command = compiler.get_exelist() + ['$ARGS', '$in']
        description = 'Compiling Vala source $in'
        self.add_rule(NinjaRule(rule, command, [], description, extra='restat = 1'))

    def generate_cython_compile_rules(self, compiler: 'Compiler') -> None:
        rule = self.compiler_to_rule_name(compiler)
        description = 'Compiling Cython source $in'
        command = compiler.get_exelist()
        depargs = compiler.get_dependency_gen_args('$out', '$DEPFILE')
        depfile = '$out.dep' if depargs else None
        args = depargs + ['$ARGS', '$in']
        args += NinjaCommandArg.list(compiler.get_output_args('$out'), Quoting.none)
        self.add_rule(NinjaRule(rule, command + args, [], description, depfile=depfile, extra='restat = 1'))

    def generate_rust_compile_rules(self, compiler):
        rule = self.compiler_to_rule_name(compiler)
        command = compiler.get_exelist() + ['$ARGS', '$in']
        description = 'Compiling Rust source $in'
        depfile = '$targetdep'
        depstyle = 'gcc'
        self.add_rule(NinjaRule(rule, command, [], description, deps=depstyle, depfile=depfile))

    def generate_swift_compile_rules(self, compiler):
        rule = self.compiler_to_rule_name(compiler)
        full_exe = self.environment.get_build_command() + ['--internal', 'dirchanger', '$RUNDIR']
        invoc = full_exe + compiler.get_exelist()
        command = invoc + ['$ARGS', '$in']
        description = 'Compiling Swift source $in'
        self.add_rule(NinjaRule(rule, command, [], description))

    def use_dyndeps_for_fortran(self) -> bool:
        """Use the new Ninja feature for scanning dependencies during build,
        rather than up front. Remove this and all old scanning code once Ninja
        minimum version is bumped to 1.10."""
        return mesonlib.version_compare(self.ninja_version, '>=1.10.0')

    def generate_fortran_dep_hack(self, crstr: str) -> None:
        if self.use_dyndeps_for_fortran():
            return
        rule = f'FORTRAN_DEP_HACK{crstr}'
        if mesonlib.is_windows():
            cmd = ['cmd', '/C']
        else:
            cmd = ['true']
        self.add_rule_comment(NinjaComment('Workaround for these issues:\nhttps://groups.google.com/forum/#!topic/ninja-build/j-2RfBIOd_8\nhttps://gcc.gnu.org/bugzilla/show_bug.cgi?id=47485'))
        self.add_rule(NinjaRule(rule, cmd, [], 'Dep hack', extra='restat = 1'))

    def generate_llvm_ir_compile_rule(self, compiler):
        if self.created_llvm_ir_rule[compiler.for_machine]:
            return
        rule = self.get_compiler_rule_name('llvm_ir', compiler.for_machine)
        command = compiler.get_exelist()
        args = ['$ARGS'] + NinjaCommandArg.list(compiler.get_output_args('$out'), Quoting.none) + compiler.get_compile_only_args() + ['$in']
        description = 'Compiling LLVM IR object $in'
        options = self._rsp_options(compiler)
        self.add_rule(NinjaRule(rule, command, args, description, **options))
        self.created_llvm_ir_rule[compiler.for_machine] = True

    def generate_compile_rule_for(self, langname, compiler):
        if langname == 'java':
            self.generate_java_compile_rule(compiler)
            return
        if langname == 'cs':
            if self.environment.machines.matches_build_machine(compiler.for_machine):
                self.generate_cs_compile_rule(compiler)
            return
        if langname == 'vala':
            self.generate_vala_compile_rules(compiler)
            return
        if langname == 'rust':
            self.generate_rust_compile_rules(compiler)
            return
        if langname == 'swift':
            if self.environment.machines.matches_build_machine(compiler.for_machine):
                self.generate_swift_compile_rules(compiler)
            return
        if langname == 'cython':
            self.generate_cython_compile_rules(compiler)
            return
        crstr = self.get_rule_suffix(compiler.for_machine)
        options = self._rsp_options(compiler)
        if langname == 'fortran':
            self.generate_fortran_dep_hack(crstr)
            options['extra'] = 'restat = 1'
        rule = self.compiler_to_rule_name(compiler)
        if langname == 'cuda':
            depargs = NinjaCommandArg.list(compiler.get_dependency_gen_args('$CUDA_ESCAPED_TARGET', '$DEPFILE'), Quoting.none)
        else:
            depargs = NinjaCommandArg.list(compiler.get_dependency_gen_args('$out', '$DEPFILE'), Quoting.none)
        command = compiler.get_exelist()
        args = ['$ARGS'] + depargs + NinjaCommandArg.list(compiler.get_output_args('$out'), Quoting.none) + compiler.get_compile_only_args() + ['$in']
        description = f'Compiling {compiler.get_display_language()} object $out'
        if compiler.get_argument_syntax() == 'msvc':
            deps = 'msvc'
            depfile = None
        else:
            deps = 'gcc'
            depfile = '$DEPFILE'
        self.add_rule(NinjaRule(rule, command, args, description, **options, deps=deps, depfile=depfile))

    def generate_pch_rule_for(self, langname, compiler):
        if langname not in {'c', 'cpp'}:
            return
        rule = self.compiler_to_pch_rule_name(compiler)
        depargs = compiler.get_dependency_gen_args('$out', '$DEPFILE')
        if compiler.get_argument_syntax() == 'msvc':
            output = []
        else:
            output = NinjaCommandArg.list(compiler.get_output_args('$out'), Quoting.none)
        if 'mwcc' in compiler.id:
            output[0].s = '-precompile'
            command = compiler.get_exelist() + ['$ARGS'] + depargs + output + ['$in']
        else:
            command = compiler.get_exelist() + ['$ARGS'] + depargs + output + compiler.get_compile_only_args() + ['$in']
        description = 'Precompiling header $in'
        if compiler.get_argument_syntax() == 'msvc':
            deps = 'msvc'
            depfile = None
        else:
            deps = 'gcc'
            depfile = '$DEPFILE'
        self.add_rule(NinjaRule(rule, command, [], description, deps=deps, depfile=depfile))

    def generate_scanner_rules(self):
        rulename = 'depscan'
        if rulename in self.ruledict:
            return
        command = self.environment.get_build_command() + ['--internal', 'depscan']
        args = ['$picklefile', '$out', '$in']
        description = 'Module scanner.'
        rule = NinjaRule(rulename, command, args, description)
        self.add_rule(rule)

    def generate_compile_rules(self):
        for for_machine in MachineChoice:
            clist = self.environment.coredata.compilers[for_machine]
            for langname, compiler in clist.items():
                if compiler.get_id() == 'clang':
                    self.generate_llvm_ir_compile_rule(compiler)
                self.generate_compile_rule_for(langname, compiler)
                self.generate_pch_rule_for(langname, compiler)
                for mode in compiler.get_modes():
                    self.generate_compile_rule_for(langname, mode)

    def generate_generator_list_rules(self, target):
        for genlist in target.get_generated_sources():
            if isinstance(genlist, (build.CustomTarget, build.CustomTargetIndex)):
                continue
            self.generate_genlist_for_target(genlist, target)

    def replace_paths(self, target, args, override_subdir=None):
        if override_subdir:
            source_target_dir = os.path.join(self.build_to_src, override_subdir)
        else:
            source_target_dir = self.get_target_source_dir(target)
        relout = self.get_target_private_dir(target)
        args = [x.replace('@SOURCE_DIR@', self.build_to_src).replace('@BUILD_DIR@', relout) for x in args]
        args = [x.replace('@CURRENT_SOURCE_DIR@', source_target_dir) for x in args]
        args = [x.replace('@SOURCE_ROOT@', self.build_to_src).replace('@BUILD_ROOT@', '.') for x in args]
        args = [x.replace('\\', '/') for x in args]
        return args

    def generate_genlist_for_target(self, genlist: build.GeneratedList, target: build.BuildTarget) -> None:
        for x in genlist.depends:
            if isinstance(x, build.GeneratedList):
                self.generate_genlist_for_target(x, target)
        generator = genlist.get_generator()
        subdir = genlist.subdir
        exe = generator.get_exe()
        infilelist = genlist.get_inputs()
        outfilelist = genlist.get_outputs()
        extra_dependencies = self.get_target_depend_files(genlist)
        for i, curfile in enumerate(infilelist):
            if len(generator.outputs) == 1:
                sole_output = os.path.join(self.get_target_private_dir(target), outfilelist[i])
            else:
                sole_output = f'{curfile}'
            infilename = curfile.rel_to_builddir(self.build_to_src, self.get_target_private_dir(target))
            base_args = generator.get_arglist(infilename)
            outfiles = genlist.get_outputs_for(curfile)
            outfiles = [os.path.join(self.get_target_private_dir(target), of) for of in outfiles]
            if generator.depfile is None:
                rulename = 'CUSTOM_COMMAND'
                args = base_args
            else:
                rulename = 'CUSTOM_COMMAND_DEP'
                depfilename = generator.get_dep_outname(infilename)
                depfile = os.path.join(self.get_target_private_dir(target), depfilename)
                args = [x.replace('@DEPFILE@', depfile) for x in base_args]
            args = [x.replace('@INPUT@', infilename).replace('@OUTPUT@', sole_output) for x in args]
            args = self.replace_outputs(args, self.get_target_private_dir(target), outfilelist)
            if len(generator.outputs) > 1:
                outfilelist = outfilelist[len(generator.outputs):]
            args = self.replace_paths(target, args, override_subdir=subdir)
            cmdlist, reason = self.as_meson_exe_cmdline(exe, self.replace_extra_args(args, genlist), capture=outfiles[0] if generator.capture else None, env=genlist.env)
            abs_pdir = os.path.join(self.environment.get_build_dir(), self.get_target_dir(target))
            os.makedirs(abs_pdir, exist_ok=True)
            elem = NinjaBuildElement(self.all_outputs, outfiles, rulename, infilename)
            elem.add_dep([self.get_target_filename(x) for x in generator.depends])
            if generator.depfile is not None:
                elem.add_item('DEPFILE', depfile)
            if len(extra_dependencies) > 0:
                elem.add_dep(extra_dependencies)
            if len(generator.outputs) == 1:
                what = f'{sole_output!r}'
            else:
                what = f'from {sole_output!r}'
            if reason:
                reason = f' (wrapped by meson {reason})'
            elem.add_item('DESC', f'Generating {what}{reason}')
            if isinstance(exe, build.BuildTarget):
                elem.add_dep(self.get_target_filename(exe))
            elem.add_item('COMMAND', cmdlist)
            self.add_build(elem)

    def scan_fortran_module_outputs(self, target):
        """
        Find all module and submodule made available in a Fortran code file.
        """
        if self.use_dyndeps_for_fortran():
            return
        compiler = None
        for lang, c in self.environment.coredata.compilers.host.items():
            if lang == 'fortran':
                compiler = c
                break
        if compiler is None:
            self.fortran_deps[target.get_basename()] = {}
            return
        modre = re.compile(FORTRAN_MODULE_PAT, re.IGNORECASE)
        submodre = re.compile(FORTRAN_SUBMOD_PAT, re.IGNORECASE)
        module_files = {}
        submodule_files = {}
        for s in target.get_sources():
            if not compiler.can_compile(s):
                continue
            filename = s.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())
            with open(filename, encoding='ascii', errors='ignore') as f:
                for line in f:
                    modmatch = modre.match(line)
                    if modmatch is not None:
                        modname = modmatch.group(1).lower()
                        if modname in module_files:
                            raise InvalidArguments(f'Namespace collision: module {modname} defined in two files {module_files[modname]} and {s}.')
                        module_files[modname] = s
                    else:
                        submodmatch = submodre.match(line)
                        if submodmatch is not None:
                            parents = submodmatch.group(1).lower().split(':')
                            submodname = parents[0] + '_' + submodmatch.group(2).lower()
                            if submodname in submodule_files:
                                raise InvalidArguments(f'Namespace collision: submodule {submodname} defined in two files {submodule_files[submodname]} and {s}.')
                            submodule_files[submodname] = s
        self.fortran_deps[target.get_basename()] = {**module_files, **submodule_files}

    def get_fortran_deps(self, compiler: FortranCompiler, src: Path, target) -> T.List[str]:
        """
        Find all module and submodule needed by a Fortran target
        """
        if self.use_dyndeps_for_fortran():
            return []
        dirname = Path(self.get_target_private_dir(target))
        tdeps = self.fortran_deps[target.get_basename()]
        srcdir = Path(self.source_dir)
        mod_files = _scan_fortran_file_deps(src, srcdir, dirname, tdeps, compiler)
        return mod_files

    def get_no_stdlib_link_args(self, target, linker):
        if hasattr(linker, 'language') and linker.language in self.build.stdlibs[target.for_machine]:
            return linker.get_no_stdlib_link_args()
        return []

    def get_compile_debugfile_args(self, compiler, target, objfile):
        if target.has_pch():
            tfilename = self.get_target_debug_filename_abs(target)
            if not tfilename:
                tfilename = self.get_target_filename_abs(target)
            return compiler.get_compile_debugfile_args(tfilename, pch=True)
        else:
            return compiler.get_compile_debugfile_args(objfile, pch=False)

    def get_link_debugfile_name(self, linker, target) -> T.Optional[str]:
        return linker.get_link_debugfile_name(self.get_target_debug_filename(target))

    def get_link_debugfile_args(self, linker, target):
        return linker.get_link_debugfile_args(self.get_target_debug_filename(target))

    def generate_llvm_ir_compile(self, target, src):
        base_proxy = target.get_options()
        compiler = get_compiler_for_source(target.compilers.values(), src)
        commands = compiler.compiler_args()
        commands += compilers.get_base_compile_args(base_proxy, compiler)
        if isinstance(src, File):
            if src.is_built:
                src_filename = os.path.join(src.subdir, src.fname)
            else:
                src_filename = src.fname
        elif os.path.isabs(src):
            src_filename = os.path.basename(src)
        else:
            src_filename = src
        obj_basename = self.canonicalize_filename(src_filename)
        rel_obj = os.path.join(self.get_target_private_dir(target), obj_basename)
        rel_obj += '.' + self.environment.machines[target.for_machine].get_object_suffix()
        commands += self.get_compile_debugfile_args(compiler, target, rel_obj)
        if isinstance(src, File) and src.is_built:
            rel_src = src.fname
        elif isinstance(src, File):
            rel_src = src.rel_to_builddir(self.build_to_src)
        else:
            raise InvalidArguments(f'Invalid source type: {src!r}')
        compiler_name = self.get_compiler_rule_name('llvm_ir', compiler.for_machine)
        element = NinjaBuildElement(self.all_outputs, rel_obj, compiler_name, rel_src)
        element.add_item('ARGS', commands)
        self.add_build(element)
        return (rel_obj, rel_src)

    @lru_cache(maxsize=None)
    def generate_inc_dir(self, compiler: 'Compiler', d: str, basedir: str, is_system: bool) -> T.Tuple['ImmutableListProtocol[str]', 'ImmutableListProtocol[str]']:
        if d not in ('', '.'):
            expdir = os.path.normpath(os.path.join(basedir, d))
        else:
            expdir = basedir
        srctreedir = os.path.normpath(os.path.join(self.build_to_src, expdir))
        sargs = compiler.get_include_args(srctreedir, is_system)
        if os.path.isdir(os.path.join(self.environment.get_build_dir(), expdir)):
            bargs = compiler.get_include_args(expdir, is_system)
        else:
            bargs = []
        return (sargs, bargs)

    def _generate_single_compile(self, target: build.BuildTarget, compiler: Compiler) -> CompilerArgs:
        commands = self._generate_single_compile_base_args(target, compiler)
        commands += self._generate_single_compile_target_args(target, compiler)
        return commands

    def _generate_single_compile_base_args(self, target: build.BuildTarget, compiler: 'Compiler') -> 'CompilerArgs':
        base_proxy = target.get_options()
        commands = compiler.compiler_args()
        commands += compiler.gnu_symbol_visibility_args(target.gnu_symbol_visibility)
        commands += compilers.get_base_compile_args(base_proxy, compiler)
        return commands

    @lru_cache(maxsize=None)
    def _generate_single_compile_target_args(self, target: build.BuildTarget, compiler: Compiler) -> ImmutableListProtocol[str]:
        commands = self.generate_basic_compiler_args(target, compiler)
        if target.implicit_include_directories:
            commands += self.get_custom_target_dir_include_args(target, compiler)
        for i in reversed(target.get_include_dirs()):
            basedir = i.get_curdir()
            for d in reversed(i.get_incdirs()):
                compile_obj, includeargs = self.generate_inc_dir(compiler, d, basedir, i.is_system)
                commands += compile_obj
                commands += includeargs
            for d in i.get_extra_build_dirs():
                commands += compiler.get_include_args(d, i.is_system)
        commands += self.escape_extra_args(target.get_extra_args(compiler.get_language()))
        if compiler.language == 'd':
            commands += compiler.get_feature_args(target.d_features, self.build_to_src)
        if target.implicit_include_directories:
            commands += self.get_source_dir_include_args(target, compiler)
        if target.implicit_include_directories:
            commands += self.get_build_dir_include_args(target, compiler)
        commands += compiler.get_include_args(self.get_target_private_dir(target), False)
        return commands

    def generate_common_compile_args_per_src_type(self, target: build.BuildTarget) -> dict[str, list[str]]:
        src_type_to_args = {}
        use_pch = self.target_uses_pch(target)
        for src_type_str in target.compilers.keys():
            compiler = target.compilers[src_type_str]
            commands = self._generate_single_compile_base_args(target, compiler)
            if use_pch and 'mw' not in compiler.id:
                commands += self.get_pch_include_args(compiler, target)
            commands += self._generate_single_compile_target_args(target, compiler)
            if use_pch and 'mw' in compiler.id:
                commands += self.get_pch_include_args(compiler, target)
            commands = commands.compiler.compiler_args(commands)
            src_type_to_args[src_type_str] = commands.to_native()
        return src_type_to_args

    def generate_single_compile(self, target: build.BuildTarget, src, is_generated: bool=False, header_deps=None, order_deps: T.Optional[T.List['mesonlib.FileOrString']]=None, extra_args: T.Optional[T.List[str]]=None, unity_sources: T.Optional[T.List[mesonlib.FileOrString]]=None) -> None:
        """
        Compiles C/C++, ObjC/ObjC++, Fortran, and D sources
        """
        header_deps = header_deps if header_deps is not None else []
        order_deps = order_deps if order_deps is not None else []
        if isinstance(src, str) and src.endswith('.h'):
            raise AssertionError(f'BUG: sources should not contain headers {src!r}')
        compiler = get_compiler_for_source(target.compilers.values(), src)
        commands = self._generate_single_compile_base_args(target, compiler)
        use_pch = self.target_uses_pch(target)
        if use_pch and 'mw' not in compiler.id:
            commands += self.get_pch_include_args(compiler, target)
        commands += self._generate_single_compile_target_args(target, compiler)
        if use_pch and 'mw' in compiler.id:
            commands += self.get_pch_include_args(compiler, target)
        commands = commands.compiler.compiler_args(commands)
        if is_generated is False:
            self.create_target_source_introspection(target, compiler, commands, [src], [], unity_sources)
        else:
            self.create_target_source_introspection(target, compiler, commands, [], [src], unity_sources)
        build_dir = self.environment.get_build_dir()
        if isinstance(src, File):
            rel_src = src.rel_to_builddir(self.build_to_src)
            if os.path.isabs(rel_src):
                if src.is_built:
                    assert rel_src.startswith(build_dir)
                    rel_src = rel_src[len(build_dir) + 1:]
        elif is_generated:
            raise AssertionError(f'BUG: broken generated source file handling for {src!r}')
        else:
            raise InvalidArguments(f'Invalid source type: {src!r}')
        obj_basename = self.object_filename_from_source(target, src)
        rel_obj = os.path.join(self.get_target_private_dir(target), obj_basename)
        dep_file = compiler.depfile_for_object(rel_obj)
        commands += self.get_compile_debugfile_args(compiler, target, rel_obj)
        if self.target_uses_pch(target):
            pchlist = target.get_pch(compiler.language)
        else:
            pchlist = []
        if not pchlist:
            pch_dep = []
        elif compiler.id == 'intel':
            pch_dep = []
        else:
            arr = []
            i = os.path.join(self.get_target_private_dir(target), compiler.get_pch_name(pchlist[0]))
            arr.append(i)
            pch_dep = arr
        compiler_name = self.compiler_to_rule_name(compiler)
        extra_deps = []
        if compiler.get_language() == 'fortran':
            if not is_generated:
                abs_src = Path(build_dir) / rel_src
                extra_deps += self.get_fortran_deps(compiler, abs_src, target)
            if not self.use_dyndeps_for_fortran():
                for modname, srcfile in self.fortran_deps[target.get_basename()].items():
                    modfile = os.path.join(self.get_target_private_dir(target), compiler.module_name_to_filename(modname))
                    if srcfile == src:
                        crstr = self.get_rule_suffix(target.for_machine)
                        depelem = NinjaBuildElement(self.all_outputs, modfile, 'FORTRAN_DEP_HACK' + crstr, rel_obj)
                        self.add_build(depelem)
            commands += compiler.get_module_outdir_args(self.get_target_private_dir(target))
        if extra_args is not None:
            commands.extend(extra_args)
        element = NinjaBuildElement(self.all_outputs, rel_obj, compiler_name, rel_src)
        self.add_header_deps(target, element, header_deps)
        for d in extra_deps:
            element.add_dep(d)
        for d in order_deps:
            if isinstance(d, File):
                d = d.rel_to_builddir(self.build_to_src)
            elif not self.has_dir_part(d):
                d = os.path.join(self.get_target_private_dir(target), d)
            element.add_orderdep(d)
        element.add_dep(pch_dep)
        for i in self.get_fortran_orderdeps(target, compiler):
            element.add_orderdep(i)
        if dep_file:
            element.add_item('DEPFILE', dep_file)
        if compiler.get_language() == 'cuda':

            def quote_make_target(targetName: str) -> str:
                result = ''
                for i, c in enumerate(targetName):
                    if c in {' ', '\t'}:
                        for j in range(i - 1, -1, -1):
                            if targetName[j] == '\\':
                                result += '\\'
                            else:
                                break
                        result += '\\'
                    elif c == '$':
                        result += '$'
                    elif c == '#':
                        result += '\\'
                    result += c
                return result
            element.add_item('CUDA_ESCAPED_TARGET', quote_make_target(rel_obj))
        element.add_item('ARGS', commands)
        self.add_dependency_scanner_entries_to_element(target, compiler, element, src)
        self.add_build(element)
        assert isinstance(rel_obj, str)
        assert isinstance(rel_src, str)
        return (rel_obj, rel_src.replace('\\', '/'))

    def add_dependency_scanner_entries_to_element(self, target: build.BuildTarget, compiler, element, src):
        if not self.should_use_dyndeps_for_target(target):
            return
        if isinstance(target, build.CompileTarget):
            return
        extension = os.path.splitext(src.fname)[1][1:]
        if extension != 'C':
            extension = extension.lower()
        if not (extension in compilers.lang_suffixes['fortran'] or extension in compilers.lang_suffixes['cpp']):
            return
        dep_scan_file = self.get_dep_scan_file_for(target)
        element.add_item('dyndep', dep_scan_file)
        element.add_orderdep(dep_scan_file)

    def get_dep_scan_file_for(self, target: build.BuildTarget) -> str:
        return os.path.join(self.get_target_private_dir(target), 'depscan.dd')

    def add_header_deps(self, target, ninja_element, header_deps):
        for d in header_deps:
            if isinstance(d, File):
                d = d.rel_to_builddir(self.build_to_src)
            elif not self.has_dir_part(d):
                d = os.path.join(self.get_target_private_dir(target), d)
            ninja_element.add_dep(d)

    def has_dir_part(self, fname: mesonlib.FileOrString) -> bool:
        if isinstance(fname, File):
            return fname.subdir != ''
        return has_path_sep(fname)

    def get_fortran_orderdeps(self, target, compiler):
        if compiler.language != 'fortran':
            return []
        return [os.path.join(self.get_target_dir(lt), lt.get_filename()) for lt in itertools.chain(target.link_targets, target.link_whole_targets)]

    def generate_msvc_pch_command(self, target, compiler, pch):
        header = pch[0]
        pchname = compiler.get_pch_name(header)
        dst = os.path.join(self.get_target_private_dir(target), pchname)
        commands = []
        commands += self.generate_basic_compiler_args(target, compiler)
        if len(pch) == 1:
            source = self.create_msvc_pch_implementation(target, compiler.get_language(), pch[0])
            pch_header_dir = os.path.dirname(os.path.join(self.build_to_src, target.get_source_subdir(), header))
            commands += compiler.get_include_args(pch_header_dir, False)
        else:
            source = os.path.join(self.build_to_src, target.get_source_subdir(), pch[1])
        just_name = os.path.basename(header)
        objname, pch_args = compiler.gen_pch_args(just_name, source, dst)
        commands += pch_args
        commands += self._generate_single_compile(target, compiler)
        commands += self.get_compile_debugfile_args(compiler, target, objname)
        dep = dst + '.' + compiler.get_depfile_suffix()
        link_objects = [objname] if compiler.should_link_pch_object() else []
        return (commands, dep, dst, link_objects, source)

    def generate_gcc_pch_command(self, target, compiler, pch):
        commands = self._generate_single_compile(target, compiler)
        if pch.split('.')[-1] == 'h' and compiler.language == 'cpp':
            commands += ['-x', 'c++-header']
        dst = os.path.join(self.get_target_private_dir(target), os.path.basename(pch) + '.' + compiler.get_pch_suffix())
        dep = dst + '.' + compiler.get_depfile_suffix()
        return (commands, dep, dst, [])

    def generate_mwcc_pch_command(self, target, compiler, pch):
        commands = self._generate_single_compile(target, compiler)
        dst = os.path.join(self.get_target_private_dir(target), os.path.basename(pch) + '.' + compiler.get_pch_suffix())
        dep = os.path.splitext(dst)[0] + '.' + compiler.get_depfile_suffix()
        return (commands, dep, dst, [])

    def generate_pch(self, target, header_deps=None):
        header_deps = header_deps if header_deps is not None else []
        pch_objects = []
        for lang in ['c', 'cpp']:
            pch = target.get_pch(lang)
            if not pch:
                continue
            if not has_path_sep(pch[0]) or not has_path_sep(pch[-1]):
                msg = f'Precompiled header of {target.get_basename()!r} must not be in the same directory as source, please put it in a subdirectory.'
                raise InvalidArguments(msg)
            compiler: Compiler = target.compilers[lang]
            if compiler.get_argument_syntax() == 'msvc':
                commands, dep, dst, objs, src = self.generate_msvc_pch_command(target, compiler, pch)
                extradep = os.path.join(self.build_to_src, target.get_source_subdir(), pch[0])
            elif compiler.id == 'intel':
                continue
            elif 'mwcc' in compiler.id:
                src = os.path.join(self.build_to_src, target.get_source_subdir(), pch[0])
                commands, dep, dst, objs = self.generate_mwcc_pch_command(target, compiler, pch[0])
                extradep = None
            else:
                src = os.path.join(self.build_to_src, target.get_source_subdir(), pch[0])
                commands, dep, dst, objs = self.generate_gcc_pch_command(target, compiler, pch[0])
                extradep = None
            pch_objects += objs
            rulename = self.compiler_to_pch_rule_name(compiler)
            elem = NinjaBuildElement(self.all_outputs, objs + [dst], rulename, src)
            if extradep is not None:
                elem.add_dep(extradep)
            self.add_header_deps(target, elem, header_deps)
            elem.add_item('ARGS', commands)
            elem.add_item('DEPFILE', dep)
            self.add_build(elem)
        return pch_objects

    def get_target_shsym_filename(self, target):
        targetdir = self.get_target_private_dir(target)
        return os.path.join(targetdir, target.get_filename() + '.symbols')

    def generate_shsym(self, target):
        target_file = self.get_target_filename(target)
        symname = self.get_target_shsym_filename(target)
        elem = NinjaBuildElement(self.all_outputs, symname, 'SHSYM', target_file)
        elem.add_item('IMPLIB', self.get_target_filename_for_linking(target))
        if self.environment.is_cross_build():
            elem.add_item('CROSS', '--cross-host=' + self.environment.machines[target.for_machine].system)
        self.add_build(elem)

    def get_import_filename(self, target):
        return os.path.join(self.get_target_dir(target), target.import_filename)

    def get_target_type_link_args(self, target, linker):
        commands = []
        if isinstance(target, build.Executable):
            commands += linker.get_std_exe_link_args()
            if target.export_dynamic:
                commands += linker.gen_export_dynamic_link_args(self.environment)
            if target.import_filename:
                commands += linker.gen_import_library_args(self.get_import_filename(target))
            if target.pie:
                commands += linker.get_pie_link_args()
            if target.vs_module_defs and hasattr(linker, 'gen_vs_module_defs_args'):
                commands += linker.gen_vs_module_defs_args(target.vs_module_defs.rel_to_builddir(self.build_to_src))
        elif isinstance(target, build.SharedLibrary):
            if isinstance(target, build.SharedModule):
                commands += linker.get_std_shared_module_link_args(target.get_options())
            else:
                commands += linker.get_std_shared_lib_link_args()
            commands += linker.get_pic_args()
            if not isinstance(target, build.SharedModule) or target.force_soname:
                commands += linker.get_soname_args(self.environment, target.prefix, target.name, target.suffix, target.soversion, target.darwin_versions)
            if target.vs_module_defs and hasattr(linker, 'gen_vs_module_defs_args'):
                commands += linker.gen_vs_module_defs_args(target.vs_module_defs.rel_to_builddir(self.build_to_src))
            if target.import_filename:
                commands += linker.gen_import_library_args(self.get_import_filename(target))
        elif isinstance(target, build.StaticLibrary):
            commands += linker.get_std_link_args(self.environment, not target.should_install())
        else:
            raise RuntimeError('Unknown build target type.')
        return commands

    def get_target_type_link_args_post_dependencies(self, target, linker):
        commands = []
        if isinstance(target, build.Executable):
            m = self.environment.machines[target.for_machine]
            if m.is_windows() or m.is_cygwin():
                commands += linker.get_win_subsystem_args(target.win_subsystem)
        return commands

    def get_link_whole_args(self, linker, target):
        use_custom = False
        if linker.id == 'msvc':
            if mesonlib.version_compare(linker.version, '<19.00.23918'):
                use_custom = True
        if use_custom:
            objects_from_static_libs: T.List[ExtractedObjects] = []
            for dep in target.link_whole_targets:
                l = dep.extract_all_objects(False)
                objects_from_static_libs += self.determine_ext_objs(l, '')
                objects_from_static_libs.extend(self.flatten_object_list(dep)[0])
            return objects_from_static_libs
        else:
            target_args = self.build_target_link_arguments(linker, target.link_whole_targets)
            return linker.get_link_whole_for(target_args) if target_args else []

    @lru_cache(maxsize=None)
    def guess_library_absolute_path(self, linker, libname, search_dirs, patterns) -> Path:
        from ..compilers.c import CCompiler
        for d in search_dirs:
            for p in patterns:
                trial = CCompiler._get_trials_from_pattern(p, d, libname)
                if not trial:
                    continue
                trial = CCompiler._get_file_from_list(self.environment, trial)
                if not trial:
                    continue
                return trial

    def guess_external_link_dependencies(self, linker, target, commands, internal):
        search_dirs = OrderedSet()
        libs = OrderedSet()
        absolute_libs = []
        build_dir = self.environment.get_build_dir()
        it = iter(linker.native_args_to_unix(commands))
        for item in it:
            if item in internal and (not item.startswith('-')):
                continue
            if item.startswith('-L'):
                if len(item) > 2:
                    path = item[2:]
                else:
                    try:
                        path = next(it)
                    except StopIteration:
                        mlog.warning('Generated linker command has -L argument without following path')
                        break
                if not os.path.isabs(path):
                    path = os.path.join(build_dir, path)
                search_dirs.add(path)
            elif item.startswith('-l'):
                if len(item) > 2:
                    lib = item[2:]
                else:
                    try:
                        lib = next(it)
                    except StopIteration:
                        mlog.warning("Generated linker command has '-l' argument without following library name")
                        break
                libs.add(lib)
            elif os.path.isabs(item) and self.environment.is_library(item) and os.path.isfile(item):
                absolute_libs.append(item)
        guessed_dependencies = []
        try:
            static_patterns = linker.get_library_naming(self.environment, LibType.STATIC, strict=True)
            shared_patterns = linker.get_library_naming(self.environment, LibType.SHARED, strict=True)
            search_dirs = tuple(search_dirs) + tuple(linker.get_library_dirs(self.environment))
            for libname in libs:
                staticlibs = self.guess_library_absolute_path(linker, libname, search_dirs, static_patterns)
                sharedlibs = self.guess_library_absolute_path(linker, libname, search_dirs, shared_patterns)
                if staticlibs:
                    guessed_dependencies.append(staticlibs.resolve().as_posix())
                if sharedlibs:
                    guessed_dependencies.append(sharedlibs.resolve().as_posix())
        except (mesonlib.MesonException, AttributeError) as e:
            if 'get_library_naming' not in str(e):
                raise
        return guessed_dependencies + absolute_libs

    def generate_prelink(self, target, obj_list):
        assert isinstance(target, build.StaticLibrary)
        prelink_name = os.path.join(self.get_target_private_dir(target), target.name + '-prelink.o')
        elem = NinjaBuildElement(self.all_outputs, [prelink_name], 'CUSTOM_COMMAND', obj_list)
        prelinker = target.get_prelinker()
        cmd = prelinker.exelist[:]
        cmd += prelinker.get_prelink_args(prelink_name, obj_list)
        cmd = self.replace_paths(target, cmd)
        elem.add_item('COMMAND', cmd)
        elem.add_item('description', f'Prelinking {prelink_name}.')
        self.add_build(elem)
        return [prelink_name]

    def generate_link(self, target: build.BuildTarget, outname, obj_list, linker: T.Union['Compiler', 'StaticLinker'], extra_args=None, stdlib_args=None):
        extra_args = extra_args if extra_args is not None else []
        stdlib_args = stdlib_args if stdlib_args is not None else []
        implicit_outs = []
        if isinstance(target, build.StaticLibrary):
            linker_base = 'STATIC'
        else:
            linker_base = linker.get_language()
        if isinstance(target, build.SharedLibrary):
            self.generate_shsym(target)
        crstr = self.get_rule_suffix(target.for_machine)
        linker_rule = linker_base + '_LINKER' + crstr
        commands = linker.compiler_args()
        if isinstance(target, build.StaticLibrary):
            commands += linker.get_base_link_args(target.get_options())
        else:
            commands += compilers.get_base_link_args(target.get_options(), linker, isinstance(target, build.SharedModule), self.environment.get_build_dir())
        commands += self.get_no_stdlib_link_args(target, linker)
        commands += linker.get_linker_always_args()
        commands += linker.get_optimization_link_args(target.get_option(OptionKey('optimization')))
        if target.get_option(OptionKey('debug')):
            commands += self.get_link_debugfile_args(linker, target)
            debugfile = self.get_link_debugfile_name(linker, target)
            if debugfile is not None:
                implicit_outs += [debugfile]
        commands += self.get_target_type_link_args(target, linker)
        if not isinstance(target, build.StaticLibrary):
            commands += self.get_link_whole_args(linker, target)
        if not isinstance(target, build.StaticLibrary):
            commands += self.build.get_project_link_args(linker, target.subproject, target.for_machine)
            commands += self.build.get_global_link_args(linker, target.for_machine)
            commands += self.environment.coredata.get_external_link_args(target.for_machine, linker.get_language())
        if has_path_sep(target.name):
            target_slashname_workaround_dir = os.path.join(os.path.dirname(target.name), self.get_target_dir(target))
        else:
            target_slashname_workaround_dir = self.get_target_dir(target)
        rpath_args, target.rpath_dirs_to_remove = linker.build_rpath_args(self.environment, self.environment.get_build_dir(), target_slashname_workaround_dir, self.determine_rpath_dirs(target), target.build_rpath, target.install_rpath)
        commands += rpath_args
        if linker_base == 'STATIC':
            dependencies = []
        else:
            dependencies = target.get_dependencies()
        internal = self.build_target_link_arguments(linker, dependencies)
        commands += internal
        if not isinstance(target, build.StaticLibrary):
            commands += linker.get_target_link_args(target)
            for dep in target.get_external_deps():
                commands.extend_preserving_lflags(linker.get_dependency_link_args(dep))
            for d in target.get_dependencies():
                if isinstance(d, build.StaticLibrary):
                    for dep in d.get_external_deps():
                        commands.extend_preserving_lflags(linker.get_dependency_link_args(dep))
        commands += self.get_target_type_link_args_post_dependencies(target, linker)
        if isinstance(linker, Compiler):
            commands += linker.get_option_link_args(target.get_options())
        dep_targets = []
        dep_targets.extend(self.guess_external_link_dependencies(linker, target, commands, internal))
        custom_target_libraries = self.get_custom_target_provided_libraries(target)
        commands += extra_args
        commands += custom_target_libraries
        commands += stdlib_args
        dep_targets.extend([self.get_dependency_filename(t) for t in dependencies])
        dep_targets.extend([self.get_dependency_filename(t) for t in target.link_depends])
        elem = NinjaBuildElement(self.all_outputs, outname, linker_rule, obj_list, implicit_outs=implicit_outs)
        elem.add_dep(dep_targets + custom_target_libraries)
        elem.add_item('LINK_ARGS', commands)
        self.create_target_linker_introspection(target, linker, commands)
        return elem

    def get_dependency_filename(self, t):
        if isinstance(t, build.SharedLibrary):
            return self.get_target_shsym_filename(t)
        elif isinstance(t, mesonlib.File):
            if t.is_built:
                return t.relative_name()
            else:
                return t.absolute_path(self.environment.get_source_dir(), self.environment.get_build_dir())
        return self.get_target_filename(t)

    def generate_shlib_aliases(self, target, outdir):
        for alias, to, tag in target.get_aliases():
            aliasfile = os.path.join(outdir, alias)
            abs_aliasfile = os.path.join(self.environment.get_build_dir(), outdir, alias)
            try:
                os.remove(abs_aliasfile)
            except Exception:
                pass
            try:
                os.symlink(to, abs_aliasfile)
            except NotImplementedError:
                mlog.debug('Library versioning disabled because symlinks are not supported.')
            except OSError:
                mlog.debug('Library versioning disabled because we do not have symlink creation privileges.')
            else:
                self.implicit_meson_outs.append(aliasfile)

    def generate_custom_target_clean(self, trees: T.List[str]) -> str:
        e = self.create_phony_target('clean-ctlist', 'CUSTOM_COMMAND', 'PHONY')
        d = CleanTrees(self.environment.get_build_dir(), trees)
        d_file = os.path.join(self.environment.get_scratch_dir(), 'cleantrees.dat')
        e.add_item('COMMAND', self.environment.get_build_command() + ['--internal', 'cleantrees', d_file])
        e.add_item('description', 'Cleaning custom target directories')
        self.add_build(e)
        with open(d_file, 'wb') as ofile:
            pickle.dump(d, ofile)
        return 'clean-ctlist'

    def generate_gcov_clean(self) -> None:
        gcno_elem = self.create_phony_target('clean-gcno', 'CUSTOM_COMMAND', 'PHONY')
        gcno_elem.add_item('COMMAND', mesonlib.get_meson_command() + ['--internal', 'delwithsuffix', '.', 'gcno'])
        gcno_elem.add_item('description', 'Deleting gcno files')
        self.add_build(gcno_elem)
        gcda_elem = self.create_phony_target('clean-gcda', 'CUSTOM_COMMAND', 'PHONY')
        gcda_elem.add_item('COMMAND', mesonlib.get_meson_command() + ['--internal', 'delwithsuffix', '.', 'gcda'])
        gcda_elem.add_item('description', 'Deleting gcda files')
        self.add_build(gcda_elem)

    def get_user_option_args(self):
        cmds = []
        for k, v in self.environment.coredata.options.items():
            if k.is_project():
                cmds.append('-D' + str(k) + '=' + (v.value if isinstance(v.value, str) else str(v.value).lower()))
        return sorted(cmds)

    def generate_dist(self) -> None:
        elem = self.create_phony_target('dist', 'CUSTOM_COMMAND', 'PHONY')
        elem.add_item('DESC', 'Creating source packages')
        elem.add_item('COMMAND', self.environment.get_build_command() + ['dist'])
        elem.add_item('pool', 'console')
        self.add_build(elem)

    def generate_scanbuild(self) -> None:
        if not environment.detect_scanbuild():
            return
        if 'scan-build' in self.all_outputs:
            return
        cmd = self.environment.get_build_command() + ['--internal', 'scanbuild', self.environment.source_dir, self.environment.build_dir, self.build.get_subproject_dir()] + self.environment.get_build_command() + ['setup'] + self.get_user_option_args()
        elem = self.create_phony_target('scan-build', 'CUSTOM_COMMAND', 'PHONY')
        elem.add_item('COMMAND', cmd)
        elem.add_item('pool', 'console')
        self.add_build(elem)

    def generate_clangtool(self, name: str, extra_arg: T.Optional[str]=None) -> None:
        target_name = 'clang-' + name
        extra_args = []
        if extra_arg:
            target_name += f'-{extra_arg}'
            extra_args.append(f'--{extra_arg}')
        if not os.path.exists(os.path.join(self.environment.source_dir, '.clang-' + name)) and (not os.path.exists(os.path.join(self.environment.source_dir, '_clang-' + name))):
            return
        if target_name in self.all_outputs:
            return
        cmd = self.environment.get_build_command() + ['--internal', 'clang' + name, self.environment.source_dir, self.environment.build_dir] + extra_args
        elem = self.create_phony_target(target_name, 'CUSTOM_COMMAND', 'PHONY')
        elem.add_item('COMMAND', cmd)
        elem.add_item('pool', 'console')
        self.add_build(elem)

    def generate_clangformat(self) -> None:
        if not environment.detect_clangformat():
            return
        self.generate_clangtool('format')
        self.generate_clangtool('format', 'check')

    def generate_clangtidy(self) -> None:
        import shutil
        if not shutil.which('clang-tidy'):
            return
        self.generate_clangtool('tidy')
        self.generate_clangtool('tidy', 'fix')

    def generate_tags(self, tool: str, target_name: str) -> None:
        import shutil
        if not shutil.which(tool):
            return
        if target_name in self.all_outputs:
            return
        cmd = self.environment.get_build_command() + ['--internal', 'tags', tool, self.environment.source_dir]
        elem = self.create_phony_target(target_name, 'CUSTOM_COMMAND', 'PHONY')
        elem.add_item('COMMAND', cmd)
        elem.add_item('pool', 'console')
        self.add_build(elem)

    def generate_utils(self) -> None:
        self.generate_scanbuild()
        self.generate_clangformat()
        self.generate_clangtidy()
        self.generate_tags('etags', 'TAGS')
        self.generate_tags('ctags', 'ctags')
        self.generate_tags('cscope', 'cscope')
        cmd = self.environment.get_build_command() + ['--internal', 'uninstall']
        elem = self.create_phony_target('uninstall', 'CUSTOM_COMMAND', 'PHONY')
        elem.add_item('COMMAND', cmd)
        elem.add_item('pool', 'console')
        self.add_build(elem)

    def generate_ending(self) -> None:
        for targ, deps in [('all', self.get_build_by_default_targets()), ('meson-test-prereq', self.get_testlike_targets()), ('meson-benchmark-prereq', self.get_testlike_targets(True))]:
            targetlist = []
            if targ == 'all':
                targetlist.extend(['meson-test-prereq', 'meson-benchmark-prereq'])
            for t in deps.values():
                if isinstance(t, build.SharedLibrary) and t.aix_so_archive:
                    if self.environment.machines[t.for_machine].is_aix():
                        linker, stdlib_args = self.determine_linker_and_stdlib_args(t)
                        t.get_outputs()[0] = linker.get_archive_name(t.get_outputs()[0])
                targetlist.append(os.path.join(self.get_target_dir(t), t.get_outputs()[0]))
            elem = NinjaBuildElement(self.all_outputs, targ, 'phony', targetlist)
            self.add_build(elem)
        elem = self.create_phony_target('clean', 'CUSTOM_COMMAND', 'PHONY')
        elem.add_item('COMMAND', self.ninja_command + ['-t', 'clean'])
        elem.add_item('description', 'Cleaning')
        ctlist = []
        for t in self.build.get_targets().values():
            if isinstance(t, build.CustomTarget):
                for o in t.get_outputs():
                    ctlist.append(os.path.join(self.get_target_dir(t), o))
        if ctlist:
            elem.add_dep(self.generate_custom_target_clean(ctlist))
        if OptionKey('b_coverage') in self.environment.coredata.options and self.environment.coredata.options[OptionKey('b_coverage')].value:
            self.generate_gcov_clean()
            elem.add_dep('clean-gcda')
            elem.add_dep('clean-gcno')
        self.add_build(elem)
        deps = self.get_regen_filelist()
        elem = NinjaBuildElement(self.all_outputs, 'build.ninja', 'REGENERATE_BUILD', deps)
        elem.add_item('pool', 'console')
        self.add_build(elem)
        if self.implicit_meson_outs:
            elem = NinjaBuildElement(self.all_outputs, 'meson-implicit-outs', 'phony', self.implicit_meson_outs)
            self.add_build(elem)
        elem = NinjaBuildElement(self.all_outputs, 'reconfigure', 'REGENERATE_BUILD', 'PHONY')
        elem.add_item('pool', 'console')
        self.add_build(elem)
        elem = NinjaBuildElement(self.all_outputs, deps, 'phony', '')
        self.add_build(elem)

    def get_introspection_data(self, target_id: str, target: build.Target) -> T.List[T.Dict[str, T.Union[bool, str, T.List[T.Union[str, T.Dict[str, T.Union[str, T.List[str], bool]]]]]]]:
        data = self.introspection_data.get(target_id)
        if not data:
            return super().get_introspection_data(target_id, target)
        return list(data.values())