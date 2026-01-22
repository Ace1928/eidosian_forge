from __future__ import annotations
import copy
import itertools
import os
import xml.dom.minidom
import xml.etree.ElementTree as ET
import uuid
import typing as T
from pathlib import Path, PurePath, PureWindowsPath
import re
from collections import Counter
from . import backends
from .. import build
from .. import mlog
from .. import compilers
from .. import mesonlib
from ..mesonlib import (
from ..environment import Environment, build_filename
from .. import coredata
def add_non_makefile_vcxproj_elements(self, root: ET.Element, type_config: ET.Element, target, platform: str, subsystem, build_args, target_args, target_defines, target_inc_dirs, file_args) -> None:
    compiler = self._get_cl_compiler(target)
    buildtype_link_args = compiler.get_optimization_link_args(self.optimization)
    down = self.target_to_build_root(target)
    ET.SubElement(type_config, 'WholeProgramOptimization').text = 'false'
    ET.SubElement(type_config, 'BasicRuntimeChecks').text = 'Default'
    if '/INCREMENTAL:NO' in buildtype_link_args:
        ET.SubElement(type_config, 'LinkIncremental').text = 'false'
    compiles = ET.SubElement(root, 'ItemDefinitionGroup')
    clconf = ET.SubElement(compiles, 'ClCompile')
    if True in (dep.name == 'openmp' for dep in target.get_external_deps()):
        ET.SubElement(clconf, 'OpenMPSupport').text = 'true'
    vscrt_type = target.get_option(OptionKey('b_vscrt'))
    vscrt_val = compiler.get_crt_val(vscrt_type, self.buildtype)
    if vscrt_val == 'mdd':
        ET.SubElement(type_config, 'UseDebugLibraries').text = 'true'
        ET.SubElement(clconf, 'RuntimeLibrary').text = 'MultiThreadedDebugDLL'
    elif vscrt_val == 'mt':
        ET.SubElement(type_config, 'UseDebugLibraries').text = 'false'
        ET.SubElement(clconf, 'RuntimeLibrary').text = 'MultiThreaded'
    elif vscrt_val == 'mtd':
        ET.SubElement(type_config, 'UseDebugLibraries').text = 'true'
        ET.SubElement(clconf, 'RuntimeLibrary').text = 'MultiThreadedDebug'
    else:
        ET.SubElement(type_config, 'UseDebugLibraries').text = 'false'
        ET.SubElement(clconf, 'RuntimeLibrary').text = 'MultiThreadedDLL'
    if '/fsanitize=address' in build_args:
        ET.SubElement(type_config, 'EnableASAN').text = 'true'
    if '/ZI' in build_args:
        ET.SubElement(clconf, 'DebugInformationFormat').text = 'EditAndContinue'
    elif '/Zi' in build_args:
        ET.SubElement(clconf, 'DebugInformationFormat').text = 'ProgramDatabase'
    elif '/Z7' in build_args:
        ET.SubElement(clconf, 'DebugInformationFormat').text = 'OldStyle'
    else:
        ET.SubElement(clconf, 'DebugInformationFormat').text = 'None'
    if '/RTC1' in build_args:
        ET.SubElement(clconf, 'BasicRuntimeChecks').text = 'EnableFastChecks'
    elif '/RTCu' in build_args:
        ET.SubElement(clconf, 'BasicRuntimeChecks').text = 'UninitializedLocalUsageCheck'
    elif '/RTCs' in build_args:
        ET.SubElement(clconf, 'BasicRuntimeChecks').text = 'StackFrameRuntimeCheck'
    if 'cpp' in target.compilers:
        eh = target.get_option(OptionKey('eh', machine=target.for_machine, lang='cpp'))
        if eh == 'a':
            ET.SubElement(clconf, 'ExceptionHandling').text = 'Async'
        elif eh == 's':
            ET.SubElement(clconf, 'ExceptionHandling').text = 'SyncCThrow'
        elif eh == 'none':
            ET.SubElement(clconf, 'ExceptionHandling').text = 'false'
        else:
            ET.SubElement(clconf, 'ExceptionHandling').text = 'Sync'
    if len(target_args) > 0:
        target_args.append('%(AdditionalOptions)')
        ET.SubElement(clconf, 'AdditionalOptions').text = ' '.join(target_args)
    ET.SubElement(clconf, 'AdditionalIncludeDirectories').text = ';'.join(target_inc_dirs)
    target_defines.append('%(PreprocessorDefinitions)')
    ET.SubElement(clconf, 'PreprocessorDefinitions').text = ';'.join(target_defines)
    ET.SubElement(clconf, 'FunctionLevelLinking').text = 'true'
    warning_level = T.cast('str', target.get_option(OptionKey('warning_level')))
    warning_level = 'EnableAllWarnings' if warning_level == 'everything' else 'Level' + str(1 + int(warning_level))
    ET.SubElement(clconf, 'WarningLevel').text = warning_level
    if target.get_option(OptionKey('werror')):
        ET.SubElement(clconf, 'TreatWarningAsError').text = 'true'
    o_flags = split_o_flags_args(build_args)
    if '/Ox' in o_flags:
        ET.SubElement(clconf, 'Optimization').text = 'Full'
    elif '/O2' in o_flags:
        ET.SubElement(clconf, 'Optimization').text = 'MaxSpeed'
    elif '/O1' in o_flags:
        ET.SubElement(clconf, 'Optimization').text = 'MinSpace'
    elif '/Od' in o_flags:
        ET.SubElement(clconf, 'Optimization').text = 'Disabled'
    if '/Oi' in o_flags:
        ET.SubElement(clconf, 'IntrinsicFunctions').text = 'true'
    if '/Ob1' in o_flags:
        ET.SubElement(clconf, 'InlineFunctionExpansion').text = 'OnlyExplicitInline'
    elif '/Ob2' in o_flags:
        ET.SubElement(clconf, 'InlineFunctionExpansion').text = 'AnySuitable'
    if '/Os' in o_flags or '/O1' in o_flags:
        ET.SubElement(clconf, 'FavorSizeOrSpeed').text = 'Size'
    elif '/Od' not in o_flags:
        ET.SubElement(clconf, 'FavorSizeOrSpeed').text = 'Speed'
    self.generate_lang_standard_info(file_args, clconf)
    resourcecompile = ET.SubElement(compiles, 'ResourceCompile')
    ET.SubElement(resourcecompile, 'PreprocessorDefinitions')
    link = ET.SubElement(compiles, 'Link')
    extra_link_args = compiler.compiler_args()
    extra_link_args += compiler.get_optimization_link_args(self.optimization)
    if self.debug:
        self.generate_debug_information(link)
    else:
        ET.SubElement(link, 'GenerateDebugInformation').text = 'false'
    if not isinstance(target, build.StaticLibrary):
        if isinstance(target, build.SharedModule):
            extra_link_args += compiler.get_std_shared_module_link_args(target.get_options())
        extra_link_args += self.build.get_project_link_args(compiler, target.subproject, target.for_machine)
        extra_link_args += self.build.get_global_link_args(compiler, target.for_machine)
        extra_link_args += self.environment.coredata.get_external_link_args(target.for_machine, compiler.get_language())
        extra_link_args += target.link_args
        for dep in target.get_external_deps():
            if dep.name == 'openmp':
                ET.SubElement(clconf, 'OpenMPSupport').text = 'true'
            else:
                extra_link_args.extend_direct(dep.get_link_args())
        for d in target.get_dependencies():
            if isinstance(d, build.StaticLibrary):
                for dep in d.get_external_deps():
                    if dep.name == 'openmp':
                        ET.SubElement(clconf, 'OpenMPSupport').text = 'true'
                    else:
                        extra_link_args.extend_direct(dep.get_link_args())
    extra_link_args += compiler.get_option_link_args(target.get_options())
    additional_libpaths, additional_links, extra_link_args = self.split_link_args(extra_link_args.to_native())
    for t in target.get_dependencies():
        if isinstance(t, build.CustomTargetIndex):
            lobj = t
        else:
            lobj = self.build.targets[t.get_id()]
        linkname = os.path.join(down, self.get_target_filename_for_linking(lobj))
        if t in target.link_whole_targets:
            if compiler.id == 'msvc' and version_compare(compiler.version, '<19.00.23918'):
                l = t.extract_all_objects(False)
                for gen in l.genlist:
                    for src in gen.get_outputs():
                        if self.environment.is_source(src):
                            path = self.get_target_generated_dir(t, gen, src)
                            gen_src_ext = '.' + os.path.splitext(path)[1][1:]
                            extra_link_args.append(path[:-len(gen_src_ext)] + '.obj')
                for src in l.srclist:
                    if self.environment.is_source(src):
                        target_private_dir = self.relpath(self.get_target_private_dir(t), self.get_target_dir(t))
                        rel_obj = self.object_filename_from_source(t, src, target_private_dir)
                        extra_link_args.append(rel_obj)
                extra_link_args.extend(self.flatten_object_list(t))
            else:
                extra_link_args += compiler.get_link_whole_for(linkname)
            trelpath = self.get_target_dir_relative_to(t, target)
            tvcxproj = os.path.join(trelpath, t.get_id() + '.vcxproj')
            tid = self.environment.coredata.target_guids[t.get_id()]
            self.add_project_reference(root, tvcxproj, tid, link_outputs=True)
            self.handled_target_deps[target.get_id()].append(t.get_id())
        elif linkname not in additional_links:
            additional_links.append(linkname)
    for lib in self.get_custom_target_provided_libraries(target):
        additional_links.append(self.relpath(lib, self.get_target_dir(target)))
    if len(extra_link_args) > 0:
        extra_link_args.append('%(AdditionalOptions)')
        ET.SubElement(link, 'AdditionalOptions').text = ' '.join(extra_link_args)
    if len(additional_libpaths) > 0:
        additional_libpaths.insert(0, '%(AdditionalLibraryDirectories)')
        ET.SubElement(link, 'AdditionalLibraryDirectories').text = ';'.join(additional_libpaths)
    if len(additional_links) > 0:
        additional_links.append('%(AdditionalDependencies)')
        ET.SubElement(link, 'AdditionalDependencies').text = ';'.join(additional_links)
    ofile = ET.SubElement(link, 'OutputFile')
    ofile.text = f'$(OutDir){target.get_filename()}'
    subsys = ET.SubElement(link, 'SubSystem')
    subsys.text = subsystem
    if isinstance(target, (build.SharedLibrary, build.Executable)) and target.get_import_filename():
        ET.SubElement(link, 'ImportLibrary').text = target.get_import_filename()
    if isinstance(target, (build.SharedLibrary, build.Executable)):
        if target.vs_module_defs:
            relpath = os.path.join(down, target.vs_module_defs.rel_to_builddir(self.build_to_src))
            ET.SubElement(link, 'ModuleDefinitionFile').text = relpath
    if self.debug:
        pdb = ET.SubElement(link, 'ProgramDataBaseFileName')
        pdb.text = f'$(OutDir){target.name}.pdb'
    targetmachine = ET.SubElement(link, 'TargetMachine')
    if target.for_machine is MachineChoice.BUILD:
        targetplatform = platform.lower()
    else:
        targetplatform = self.platform.lower()
    if targetplatform == 'win32':
        targetmachine.text = 'MachineX86'
    elif targetplatform == 'x64' or detect_microsoft_gdk(targetplatform):
        targetmachine.text = 'MachineX64'
    elif targetplatform == 'arm':
        targetmachine.text = 'MachineARM'
    elif targetplatform == 'arm64':
        targetmachine.text = 'MachineARM64'
    elif targetplatform == 'arm64ec':
        targetmachine.text = 'MachineARM64EC'
    else:
        raise MesonException('Unsupported Visual Studio target machine: ' + targetplatform)
    ET.SubElement(link, 'SuppressStartupBanner').text = 'true'
    if not target.get_option(OptionKey('debug')):
        ET.SubElement(link, 'SetChecksum').text = 'true'