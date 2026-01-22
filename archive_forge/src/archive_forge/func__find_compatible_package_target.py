from __future__ import annotations
from .base import ExternalDependency, DependencyException, DependencyTypeName
from .pkgconfig import PkgConfigDependency
from ..mesonlib import (Popen_safe, OptionKey, join_args, version_compare)
from ..programs import ExternalProgram
from .. import mlog
import re
import os
import json
import typing as T
def _find_compatible_package_target(self, jdesc: T.Dict[str, str], jpack: T.Dict[str, str], dub_comp_id: str) -> T.Tuple[str, T.Set[str]]:
    dub_build_path = os.path.join(jpack['path'], '.dub', 'build')
    if not os.path.exists(dub_build_path):
        return (None, None)
    conf = jpack['configuration']
    build_type = jdesc['buildType']
    platforms = jdesc['platform']
    archs = jdesc['architecture']
    comp_versions = []
    if dub_comp_id != 'gdc':
        comp_versions.append(self.compiler.version)
        ret, res = self._call_compbin(['--version'])[0:2]
        if ret != 0:
            mlog.error('Failed to run {!r}', mlog.bold(dub_comp_id))
            return (None, None)
        d_ver_reg = re.search('v[0-9].[0-9][0-9][0-9].[0-9]', res)
        if d_ver_reg is not None:
            frontend_version = d_ver_reg.group()
            frontend_id = frontend_version.rsplit('.', 1)[0].replace('v', '').replace('.', '')
            comp_versions.extend([frontend_version, frontend_id])
    compatibilities: T.Set[str] = set()
    check_list = ('configuration', 'platform', 'arch', 'compiler', 'compiler_version')
    for entry in os.listdir(dub_build_path):
        target = os.path.join(dub_build_path, entry, jpack['targetFileName'])
        if not os.path.exists(target):
            mlog.debug('WARNING: Could not find a Dub target: ' + target)
            continue
        comps = set()
        if conf in entry:
            comps.add('configuration')
        if build_type in entry:
            comps.add('build_type')
        if all((platform in entry for platform in platforms)):
            comps.add('platform')
        if all((arch in entry for arch in archs)):
            comps.add('arch')
        if dub_comp_id in entry:
            comps.add('compiler')
        if dub_comp_id == 'gdc' or any((cv in entry for cv in comp_versions)):
            comps.add('compiler_version')
        if all((key in comps for key in check_list)):
            return (target, comps)
        else:
            compatibilities = set.union(compatibilities, comps)
    return (None, compatibilities)