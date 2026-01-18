from __future__ import annotations
import importlib.util
import inspect
import itertools
import os
import platform
import re
import sys
import sysconfig
import traceback
from types import FrameType, ModuleType
from typing import (
from coverage import env
from coverage.disposition import FileDisposition, disposition_init
from coverage.exceptions import CoverageException, PluginError
from coverage.files import TreeMatcher, GlobMatcher, ModuleMatcher
from coverage.files import prep_patterns, find_python_files, canonical_filename
from coverage.misc import sys_modules_saved
from coverage.python import source_for_file, source_for_morf
from coverage.types import TFileDisposition, TMorf, TWarnFn, TDebugCtl
def warn_already_imported_files(self) -> None:
    """Warn if files have already been imported that we will be measuring."""
    if self.include or self.source or self.source_pkgs:
        warned = set()
        for mod in list(sys.modules.values()):
            filename = getattr(mod, '__file__', None)
            if filename is None:
                continue
            if filename in warned:
                continue
            if len(getattr(mod, '__path__', ())) > 1:
                continue
            disp = self.should_trace(filename)
            if disp.has_dynamic_filename:
                continue
            if disp.trace:
                msg = f'Already imported a file that will be measured: {filename}'
                self.warn(msg, slug='already-imported')
                warned.add(filename)
            elif self.debug and self.debug.should('trace'):
                self.debug.write("Didn't trace already imported file {!r}: {}".format(disp.original_filename, disp.reason))