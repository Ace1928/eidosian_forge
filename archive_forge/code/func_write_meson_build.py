from __future__ import annotations
import os
import errno
import shutil
import subprocess
import sys
from pathlib import Path
from ._backend import Backend
from string import Template
from itertools import chain
import warnings
def write_meson_build(self, build_dir: Path) -> None:
    """Writes the meson build file at specified location"""
    meson_template = MesonTemplate(self.modulename, self.sources, self.dependencies, self.libraries, self.library_dirs, self.include_dirs, self.extra_objects, self.flib_flags, self.fc_flags, self.build_type, sys.executable)
    src = meson_template.generate_meson_build()
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    meson_build_file = Path(build_dir) / 'meson.build'
    meson_build_file.write_text(src)
    return meson_build_file