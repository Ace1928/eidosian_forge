from __future__ import annotations
import collections
import datetime
import functools
import json
import os
import re
import shutil
import string
from dataclasses import dataclass
from typing import Any, Iterable, TYPE_CHECKING, cast
import coverage
from coverage.data import CoverageData, add_data_to_hash
from coverage.exceptions import NoDataError
from coverage.files import flat_rootname
from coverage.misc import ensure_dir, file_be_gone, Hasher, isolate_module, format_local_datetime
from coverage.misc import human_sorted, plural, stdout_link
from coverage.report_core import get_analysis_to_report
from coverage.results import Analysis, Numbers
from coverage.templite import Templite
from coverage.types import TLineNo, TMorf
from coverage.version import __url__
class IncrementalChecker:
    """Logic and data to support incremental reporting."""
    STATUS_FILE = 'status.json'
    STATUS_FORMAT = 2
    NOTE = 'This file is an internal implementation detail to speed up HTML report' + ' generation. Its format can change at any time. You might be looking' + ' for the JSON report: https://coverage.rtfd.io/cmd.html#cmd-json'

    def __init__(self, directory: str) -> None:
        self.directory = directory
        self.reset()

    def reset(self) -> None:
        """Initialize to empty. Causes all files to be reported."""
        self.globals = ''
        self.files: dict[str, FileInfoDict] = {}

    def read(self) -> None:
        """Read the information we stored last time."""
        usable = False
        try:
            status_file = os.path.join(self.directory, self.STATUS_FILE)
            with open(status_file) as fstatus:
                status = json.load(fstatus)
        except (OSError, ValueError):
            usable = False
        else:
            usable = True
            if status['format'] != self.STATUS_FORMAT:
                usable = False
            elif status['version'] != coverage.__version__:
                usable = False
        if usable:
            self.files = {}
            for filename, fileinfo in status['files'].items():
                fileinfo['index']['nums'] = Numbers(*fileinfo['index']['nums'])
                self.files[filename] = fileinfo
            self.globals = status['globals']
        else:
            self.reset()

    def write(self) -> None:
        """Write the current status."""
        status_file = os.path.join(self.directory, self.STATUS_FILE)
        files = {}
        for filename, fileinfo in self.files.items():
            index = fileinfo['index']
            index['nums'] = index['nums'].init_args()
            files[filename] = fileinfo
        status = {'note': self.NOTE, 'format': self.STATUS_FORMAT, 'version': coverage.__version__, 'globals': self.globals, 'files': files}
        with open(status_file, 'w') as fout:
            json.dump(status, fout, separators=(',', ':'))

    def check_global_data(self, *data: Any) -> None:
        """Check the global data that can affect incremental reporting."""
        m = Hasher()
        for d in data:
            m.update(d)
        these_globals = m.hexdigest()
        if self.globals != these_globals:
            self.reset()
            self.globals = these_globals

    def can_skip_file(self, data: CoverageData, fr: FileReporter, rootname: str) -> bool:
        """Can we skip reporting this file?

        `data` is a CoverageData object, `fr` is a `FileReporter`, and
        `rootname` is the name being used for the file.
        """
        m = Hasher()
        m.update(fr.source().encode('utf-8'))
        add_data_to_hash(data, fr.filename, m)
        this_hash = m.hexdigest()
        that_hash = self.file_hash(rootname)
        if this_hash == that_hash:
            return True
        else:
            self.set_file_hash(rootname, this_hash)
            return False

    def file_hash(self, fname: str) -> str:
        """Get the hash of `fname`'s contents."""
        return self.files.get(fname, {}).get('hash', '')

    def set_file_hash(self, fname: str, val: str) -> None:
        """Set the hash of `fname`'s contents."""
        self.files.setdefault(fname, {})['hash'] = val

    def index_info(self, fname: str) -> IndexInfoDict:
        """Get the information for index.html for `fname`."""
        return self.files.get(fname, {}).get('index', {})

    def set_index_info(self, fname: str, info: IndexInfoDict) -> None:
        """Set the information for index.html for `fname`."""
        self.files.setdefault(fname, {})['index'] = info