import json
import os
import subprocess
import sys
from typing import List, Tuple
from pathlib import Path
from . import (METATYPES_JSON_SUFFIX, PROJECT_FILE_SUFFIX, qt_metatype_json_dir,
class QmlProjectData:
    """QML relevant project data."""

    def __init__(self):
        self._import_name: str = ''
        self._import_major_version: int = 0
        self._import_minor_version: int = 0
        self._qt_modules: List[str] = []

    def registrar_options(self):
        result = ['--import-name', self._import_name, '--major-version', str(self._import_major_version), '--minor-version', str(self._import_minor_version)]
        if self._qt_modules:
            foreign_files: List[str] = []
            meta_dir = qt_metatype_json_dir()
            for mod in self._qt_modules:
                mod_id = mod[2:].lower()
                pattern = f'qt6{mod_id}_*'
                if sys.platform != 'win32':
                    pattern += '_'
                pattern += METATYPES_JSON_SUFFIX
                for f in meta_dir.glob(pattern):
                    foreign_files.append(os.fspath(f))
                    break
                if foreign_files:
                    foreign_files_str = ','.join(foreign_files)
                    result.append(f'--foreign-types={foreign_files_str}')
        return result

    @property
    def import_name(self):
        return self._import_name

    @import_name.setter
    def import_name(self, n):
        self._import_name = n

    @property
    def import_major_version(self):
        return self._import_major_version

    @import_major_version.setter
    def import_major_version(self, v):
        self._import_major_version = v

    @property
    def import_minor_version(self):
        return self._import_minor_version

    @import_minor_version.setter
    def import_minor_version(self, v):
        self._import_minor_version = v

    @property
    def qt_modules(self):
        return self._qt_modules

    @qt_modules.setter
    def qt_modules(self, v):
        self._qt_modules = v

    def __str__(self) -> str:
        vmaj = self._import_major_version
        vmin = self._import_minor_version
        return f'"{self._import_name}" v{vmaj}.{vmin}'

    def __bool__(self) -> bool:
        return len(self._import_name) > 0 and self._import_major_version > 0