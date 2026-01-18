from __future__ import annotations
from contextlib import redirect_stdout
import collections
import dataclasses
import json
import os
from pathlib import Path, PurePath
import sys
import typing as T
from . import build, mesonlib, coredata as cdata
from .ast import IntrospectionInterpreter, BUILD_TARGET_FUNCTIONS, AstConditionLevel, AstIDGenerator, AstIndentationGenerator, AstJSONPrinter
from .backend import backends
from .dependencies import Dependency
from . import environment
from .interpreterbase import ObjectHolder
from .mesonlib import OptionKey
from .mparser import FunctionNode, ArrayNode, ArgumentNode, BaseStringNode
def write_meson_info_file(builddata: build.Build, errors: list, build_files_updated: bool=False) -> None:
    info_dir = builddata.environment.info_dir
    info_file = get_meson_info_file(info_dir)
    intro_types = get_meson_introspection_types()
    intro_info = {}
    for i, v in intro_types.items():
        if not v.func:
            continue
        intro_info[i] = {'file': f'intro-{i}.json', 'updated': i in updated_introspection_files}
    info_data = {'meson_version': split_version_string(cdata.version), 'directories': {'source': builddata.environment.get_source_dir(), 'build': builddata.environment.get_build_dir(), 'info': info_dir}, 'introspection': {'version': split_version_string(get_meson_introspection_version()), 'information': intro_info}, 'build_files_updated': build_files_updated}
    if errors:
        info_data['error'] = True
        info_data['error_list'] = [x if isinstance(x, str) else str(x) for x in errors]
    else:
        info_data['error'] = False
    tmp_file = os.path.join(info_dir, 'tmp_dump.json')
    with open(tmp_file, 'w', encoding='utf-8') as fp:
        json.dump(info_data, fp)
        fp.flush()
    os.replace(tmp_file, info_file)