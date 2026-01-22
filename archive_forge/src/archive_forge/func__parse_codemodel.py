from __future__ import annotations
from .common import CMakeException, CMakeBuildFile, CMakeConfiguration
import typing as T
from .. import mlog
from pathlib import Path
import json
import re
def _parse_codemodel(self, data: T.Dict[str, T.Any]) -> None:
    assert 'configurations' in data
    assert 'paths' in data
    source_dir = data['paths']['source']
    build_dir = data['paths']['build']

    def helper_parse_dir(dir_entry: T.Dict[str, T.Any]) -> T.Tuple[Path, Path]:
        src_dir = Path(dir_entry.get('source', '.'))
        bld_dir = Path(dir_entry.get('build', '.'))
        src_dir = src_dir if src_dir.is_absolute() else source_dir / src_dir
        bld_dir = bld_dir if bld_dir.is_absolute() else build_dir / bld_dir
        src_dir = src_dir.resolve()
        bld_dir = bld_dir.resolve()
        return (src_dir, bld_dir)

    def parse_sources(comp_group: T.Dict[str, T.Any], tgt: T.Dict[str, T.Any]) -> T.Tuple[T.List[Path], T.List[Path], T.List[int]]:
        gen = []
        src = []
        idx = []
        src_list_raw = tgt.get('sources', [])
        for i in comp_group.get('sourceIndexes', []):
            if i >= len(src_list_raw) or 'path' not in src_list_raw[i]:
                continue
            if src_list_raw[i].get('isGenerated', False):
                gen += [Path(src_list_raw[i]['path'])]
            else:
                src += [Path(src_list_raw[i]['path'])]
            idx += [i]
        return (src, gen, idx)

    def parse_target(tgt: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        src_dir, bld_dir = helper_parse_dir(cnf.get('paths', {}))
        install_paths = []
        if 'install' in tgt:
            prefix = Path(tgt['install']['prefix']['path'])
            install_paths = [prefix / x['path'] for x in tgt['install']['destinations']]
            install_paths = list(set(install_paths))
        link_flags = []
        link_libs = []
        for i in tgt.get('link', {}).get('commandFragments', []):
            if i['role'] == 'flags':
                link_flags += [i['fragment']]
            elif i['role'] == 'libraries':
                link_libs += [i['fragment']]
            elif i['role'] == 'libraryPath':
                link_flags += ['-L{}'.format(i['fragment'])]
            elif i['role'] == 'frameworkPath':
                link_flags += ['-F{}'.format(i['fragment'])]
        for i in tgt.get('archive', {}).get('commandFragments', []):
            if i['role'] == 'flags':
                link_flags += [i['fragment']]
        tgt_data = {'artifacts': [Path(x.get('path', '')) for x in tgt.get('artifacts', [])], 'sourceDirectory': src_dir, 'buildDirectory': bld_dir, 'name': tgt.get('name', ''), 'fullName': tgt.get('nameOnDisk', ''), 'hasInstallRule': 'install' in tgt, 'installPaths': install_paths, 'linkerLanguage': tgt.get('link', {}).get('language', 'CXX'), 'linkLibraries': ' '.join(link_libs), 'linkFlags': ' '.join(link_flags), 'type': tgt.get('type', 'EXECUTABLE'), 'fileGroups': []}
        processed_src_idx = []
        for cg in tgt.get('compileGroups', []):
            flags = []
            for i in cg.get('compileCommandFragments', []):
                flags += [i['fragment']]
            cg_data = {'defines': [x.get('define', '') for x in cg.get('defines', [])], 'compileFlags': ' '.join(flags), 'language': cg.get('language', 'C'), 'isGenerated': None, 'sources': [], 'includePath': cg.get('includes', [])}
            normal_src, generated_src, src_idx = parse_sources(cg, tgt)
            if normal_src:
                cg_data = dict(cg_data)
                cg_data['isGenerated'] = False
                cg_data['sources'] = normal_src
                tgt_data['fileGroups'] += [cg_data]
            if generated_src:
                cg_data = dict(cg_data)
                cg_data['isGenerated'] = True
                cg_data['sources'] = generated_src
                tgt_data['fileGroups'] += [cg_data]
            processed_src_idx += src_idx
        normal_src = []
        generated_src = []
        for idx, src in enumerate(tgt.get('sources', [])):
            if idx in processed_src_idx:
                continue
            if src.get('isGenerated', False):
                generated_src += [src['path']]
            else:
                normal_src += [src['path']]
        if normal_src:
            tgt_data['fileGroups'] += [{'isGenerated': False, 'sources': normal_src}]
        if generated_src:
            tgt_data['fileGroups'] += [{'isGenerated': True, 'sources': generated_src}]
        return tgt_data

    def parse_project(pro: T.Dict[str, T.Any]) -> T.Dict[str, T.Any]:
        p_src_dir = source_dir
        p_bld_dir = build_dir
        try:
            p_src_dir, p_bld_dir = helper_parse_dir(cnf['directories'][pro['directoryIndexes'][0]])
        except (IndexError, KeyError):
            pass
        pro_data = {'name': pro.get('name', ''), 'sourceDirectory': p_src_dir, 'buildDirectory': p_bld_dir, 'targets': []}
        for ref in pro.get('targetIndexes', []):
            tgt = {}
            try:
                tgt = cnf['targets'][ref]
            except (IndexError, KeyError):
                pass
            pro_data['targets'] += [parse_target(tgt)]
        return pro_data
    for cnf in data.get('configurations', []):
        cnf_data = {'name': cnf.get('name', ''), 'projects': []}
        for pro in cnf.get('projects', []):
            cnf_data['projects'] += [parse_project(pro)]
        self.cmake_configurations += [CMakeConfiguration(cnf_data)]