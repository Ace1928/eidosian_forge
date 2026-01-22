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
def _convert_manifest(raw_manifest: manifest.Manifest, subdir: str, path: str='') -> Manifest:
    lib = _fixup_raw_mappings(raw_manifest.get('lib', {}))
    lib.setdefault('name', raw_manifest['package']['name'])
    pkg = T.cast('manifest.FixedPackage', {fixup_meson_varname(k): v for k, v in raw_manifest['package'].items()})
    return Manifest(Package(**pkg), {k: Dependency.from_raw(v) for k, v in raw_manifest.get('dependencies', {}).items()}, {k: Dependency.from_raw(v) for k, v in raw_manifest.get('dev-dependencies', {}).items()}, {k: Dependency.from_raw(v) for k, v in raw_manifest.get('build-dependencies', {}).items()}, Library(**lib), [Binary(**_fixup_raw_mappings(b)) for b in raw_manifest.get('bin', {})], [Test(**_fixup_raw_mappings(b)) for b in raw_manifest.get('test', {})], [Benchmark(**_fixup_raw_mappings(b)) for b in raw_manifest.get('bench', {})], [Example(**_fixup_raw_mappings(b)) for b in raw_manifest.get('example', {})], raw_manifest.get('features', {}), {k: {k2: Dependency.from_raw(v2) for k2, v2 in v.get('dependencies', {}).items()} for k, v in raw_manifest.get('target', {}).items()}, subdir, path)