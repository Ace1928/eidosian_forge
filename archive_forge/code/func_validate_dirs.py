from __future__ import annotations
import argparse, datetime, glob, json, os, platform, shutil, sys, tempfile, time
import cProfile as profile
from pathlib import Path
import typing as T
from . import build, coredata, environment, interpreter, mesonlib, mintro, mlog
from .mesonlib import MesonException
def validate_dirs(self) -> T.Tuple[str, str]:
    src_dir, build_dir = self.validate_core_dirs(self.options.builddir, self.options.sourcedir)
    if Path(build_dir) in Path(src_dir).parents:
        raise MesonException(f'Build directory {build_dir} cannot be a parent of source directory {src_dir}')
    if not os.listdir(build_dir):
        self.add_vcs_ignore_files(build_dir)
        return (src_dir, build_dir)
    priv_dir = os.path.join(build_dir, 'meson-private')
    has_valid_build = os.path.exists(os.path.join(priv_dir, 'coredata.dat'))
    has_partial_build = os.path.isdir(priv_dir)
    if has_valid_build:
        if not self.options.reconfigure and (not self.options.wipe):
            print('Directory already configured.\n\nJust run your build command (e.g. ninja) and Meson will regenerate as necessary.\nRun "meson setup --reconfigure to force Meson to regenerate.\n\nIf build failures persist, run "meson setup --wipe" to rebuild from scratch\nusing the same options as passed when configuring the build.')
            if self.options.cmd_line_options:
                from . import mconf
                raise SystemExit(mconf.run_impl(self.options, build_dir))
            raise SystemExit(0)
    elif not has_partial_build and self.options.wipe:
        raise MesonException(f'Directory is not empty and does not contain a previous build:\n{build_dir}')
    return (src_dir, build_dir)