from __future__ import annotations
import dataclasses
import glob
import importlib
import itertools
import json
import os
import shutil
import collections
import typing as T
from . import builder
from . import version
from ..mesonlib import MesonException, Popen_safe, OptionKey
from .. import coredata
def _create_features(cargo: Manifest, build: builder.Builder) -> T.List[mparser.BaseNode]:
    ast: T.List[mparser.BaseNode] = []
    ast.append(build.assign(build.dict({}), 'features'))
    for depname in cargo.dependencies:
        ast.append(build.assign(build.dict({}), _options_varname(depname)))
    ast.append(build.assign(build.dict({}), 'required_deps'))
    for feature in cargo.features:
        features, dep_features, required_deps = _process_feature(cargo, feature)
        lines: T.List[mparser.BaseNode] = [build.plusassign(build.dict({build.string(d): build.bool(True) for d in required_deps}), 'required_deps'), build.plusassign(build.dict({build.string(f): build.bool(True) for f in features}), 'features')]
        for depname, enabled_features in dep_features.items():
            lines.append(build.plusassign(build.dict({build.string(_option_name(f)): build.bool(True) for f in enabled_features}), _options_varname(depname)))
        ast.append(build.if_(build.function('get_option', [build.string(_option_name(feature))]), build.block(lines)))
    ast.append(build.function('message', [build.string('Enabled features:'), build.method('keys', build.identifier('features'))]))
    return ast