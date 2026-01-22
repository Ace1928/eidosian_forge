from __future__ import annotations
from .. import mlog
import contextlib
from dataclasses import dataclass
import urllib.request
import urllib.error
import urllib.parse
import os
import hashlib
import shutil
import tempfile
import stat
import subprocess
import sys
import configparser
import time
import typing as T
import textwrap
import json
from base64 import b64encode
from netrc import netrc
from pathlib import Path, PurePath
from functools import lru_cache
from . import WrapMode
from .. import coredata
from ..mesonlib import quiet_git, GIT, ProgressBar, MesonException, windows_proof_rmtree, Popen_safe
from ..interpreterbase import FeatureNew
from ..interpreterbase import SubProject
from .. import mesonlib
class PackageDefinition:

    def __init__(self, fname: str, subproject: str=''):
        self.filename = fname
        self.subproject = SubProject(subproject)
        self.type: T.Optional[str] = None
        self.values: T.Dict[str, str] = {}
        self.provided_deps: T.Dict[str, T.Optional[str]] = {}
        self.provided_programs: T.List[str] = []
        self.diff_files: T.List[Path] = []
        self.basename = os.path.basename(fname)
        self.has_wrap = self.basename.endswith('.wrap')
        self.name = self.basename[:-5] if self.has_wrap else self.basename
        self.provided_deps[self.name.lower()] = None
        self.original_filename = fname
        self.redirected = False
        if self.has_wrap:
            self.parse_wrap()
            with open(fname, 'r', encoding='utf-8') as file:
                self.wrapfile_hash = hashlib.sha256(file.read().encode('utf-8')).hexdigest()
        self.directory = self.values.get('directory', self.name)
        if os.path.dirname(self.directory):
            raise WrapException('Directory key must be a name and not a path')
        if self.type and self.type not in ALL_TYPES:
            raise WrapException(f'Unknown wrap type {self.type!r}')
        self.filesdir = os.path.join(os.path.dirname(self.filename), 'packagefiles')

    def parse_wrap(self) -> None:
        try:
            config = configparser.ConfigParser(interpolation=None)
            config.read(self.filename, encoding='utf-8')
        except configparser.Error as e:
            raise WrapException(f'Failed to parse {self.basename}: {e!s}')
        self.parse_wrap_section(config)
        if self.type == 'redirect':
            dirname = Path(self.filename).parent
            fname = Path(self.values['filename'])
            for i, p in enumerate(fname.parts):
                if i % 2 == 0:
                    if p == '..':
                        raise WrapException('wrap-redirect filename cannot contain ".."')
                elif p != 'subprojects':
                    raise WrapException('wrap-redirect filename must be in the form foo/subprojects/bar.wrap')
            if fname.suffix != '.wrap':
                raise WrapException('wrap-redirect filename must be a .wrap file')
            fname = dirname / fname
            if not fname.is_file():
                raise WrapException(f'wrap-redirect {fname} filename does not exist')
            self.filename = str(fname)
            self.parse_wrap()
            self.redirected = True
        else:
            self.parse_provide_section(config)
        if 'patch_directory' in self.values:
            FeatureNew('Wrap files with patch_directory', '0.55.0').use(self.subproject)
        for what in ['patch', 'source']:
            if f'{what}_filename' in self.values and f'{what}_url' not in self.values:
                FeatureNew(f'Local wrap patch files without {what}_url', '0.55.0').use(self.subproject)

    def parse_wrap_section(self, config: configparser.ConfigParser) -> None:
        if len(config.sections()) < 1:
            raise WrapException(f'Missing sections in {self.basename}')
        self.wrap_section = config.sections()[0]
        if not self.wrap_section.startswith('wrap-'):
            raise WrapException(f'{self.wrap_section!r} is not a valid first section in {self.basename}')
        self.type = self.wrap_section[5:]
        self.values = dict(config[self.wrap_section])
        if 'diff_files' in self.values:
            FeatureNew('Wrap files with diff_files', '0.63.0').use(self.subproject)
            for s in self.values['diff_files'].split(','):
                path = Path(s.strip())
                if path.is_absolute():
                    raise WrapException('diff_files paths cannot be absolute')
                if '..' in path.parts:
                    raise WrapException('diff_files paths cannot contain ".."')
                self.diff_files.append(path)

    def parse_provide_section(self, config: configparser.ConfigParser) -> None:
        if config.has_section('provides'):
            raise WrapException('Unexpected "[provides]" section, did you mean "[provide]"?')
        if config.has_section('provide'):
            for k, v in config['provide'].items():
                if k == 'dependency_names':
                    names_dict = {n.strip().lower(): None for n in v.split(',')}
                    self.provided_deps.update(names_dict)
                    continue
                if k == 'program_names':
                    names_list = [n.strip() for n in v.split(',')]
                    self.provided_programs += names_list
                    continue
                if not v:
                    m = f'Empty dependency variable name for {k!r} in {self.basename}. If the subproject uses meson.override_dependency() it can be added in the "dependency_names" special key.'
                    raise WrapException(m)
                self.provided_deps[k] = v

    def get(self, key: str) -> str:
        try:
            return self.values[key]
        except KeyError:
            raise WrapException(f'Missing key {key!r} in {self.basename}')

    def get_hashfile(self, subproject_directory: str) -> str:
        return os.path.join(subproject_directory, '.meson-subproject-wrap-hash.txt')

    def update_hash_cache(self, subproject_directory: str) -> None:
        if self.has_wrap:
            with open(self.get_hashfile(subproject_directory), 'w', encoding='utf-8') as file:
                file.write(self.wrapfile_hash + '\n')