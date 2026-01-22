from __future__ import annotations
import collections
import contextlib
import os
import platform
import pstats
import re
import sys
from . import config
from .util import gc_collect
from ..util import has_compiled_ext
class ProfileStatsFile:
    """Store per-platform/fn profiling results in a file.

    There was no json module available when this was written, but now
    the file format which is very deterministically line oriented is kind of
    handy in any case for diffs and merges.

    """

    def __init__(self, filename, sort='cumulative', dump=None):
        self.force_write = config.options is not None and config.options.force_write_profiles
        self.write = self.force_write or (config.options is not None and config.options.write_profiles)
        self.fname = os.path.abspath(filename)
        self.short_fname = os.path.split(self.fname)[-1]
        self.data = collections.defaultdict(lambda: collections.defaultdict(dict))
        self.dump = dump
        self.sort = sort
        self._read()
        if self.write:
            self._write()

    @property
    def platform_key(self):
        dbapi_key = config.db.name + '_' + config.db.driver
        if config.db.name == 'sqlite' and config.db.dialect._is_url_file_db(config.db.url):
            dbapi_key += '_file'
        py_version = '.'.join([str(v) for v in sys.version_info[0:2]])
        platform_tokens = [platform.machine(), platform.system().lower(), platform.python_implementation().lower(), py_version, dbapi_key]
        platform_tokens.append('dbapiunicode')
        _has_cext = has_compiled_ext()
        platform_tokens.append(_has_cext and 'cextensions' or 'nocextensions')
        return '_'.join(platform_tokens)

    def has_stats(self):
        test_key = _current_test
        return test_key in self.data and self.platform_key in self.data[test_key]

    def result(self, callcount):
        test_key = _current_test
        per_fn = self.data[test_key]
        per_platform = per_fn[self.platform_key]
        if 'counts' not in per_platform:
            per_platform['counts'] = counts = []
        else:
            counts = per_platform['counts']
        if 'current_count' not in per_platform:
            per_platform['current_count'] = current_count = 0
        else:
            current_count = per_platform['current_count']
        has_count = len(counts) > current_count
        if not has_count:
            counts.append(callcount)
            if self.write:
                self._write()
            result = None
        else:
            result = (per_platform['lineno'], counts[current_count])
        per_platform['current_count'] += 1
        return result

    def reset_count(self):
        test_key = _current_test
        if test_key not in self.data:
            return
        per_fn = self.data[test_key]
        if self.platform_key not in per_fn:
            return
        per_platform = per_fn[self.platform_key]
        if 'counts' in per_platform:
            per_platform['counts'][:] = []

    def replace(self, callcount):
        test_key = _current_test
        per_fn = self.data[test_key]
        per_platform = per_fn[self.platform_key]
        counts = per_platform['counts']
        current_count = per_platform['current_count']
        if current_count < len(counts):
            counts[current_count - 1] = callcount
        else:
            counts[-1] = callcount
        if self.write:
            self._write()

    def _header(self):
        return "# %s\n# This file is written out on a per-environment basis.\n# For each test in aaa_profiling, the corresponding function and \n# environment is located within this file.  If it doesn't exist,\n# the test is skipped.\n# If a callcount does exist, it is compared to what we received. \n# assertions are raised if the counts do not match.\n# \n# To add a new callcount test, apply the function_call_count \n# decorator and re-run the tests using the --write-profiles \n# option - this file will be rewritten including the new count.\n# \n" % self.fname

    def _read(self):
        try:
            profile_f = open(self.fname)
        except OSError:
            return
        for lineno, line in enumerate(profile_f):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            test_key, platform_key, counts = line.split()
            per_fn = self.data[test_key]
            per_platform = per_fn[platform_key]
            c = [int(count) for count in counts.split(',')]
            per_platform['counts'] = c
            per_platform['lineno'] = lineno + 1
            per_platform['current_count'] = 0
        profile_f.close()

    def _write(self):
        print('Writing profile file %s' % self.fname)
        profile_f = open(self.fname, 'w')
        profile_f.write(self._header())
        for test_key in sorted(self.data):
            per_fn = self.data[test_key]
            profile_f.write('\n# TEST: %s\n\n' % test_key)
            for platform_key in sorted(per_fn):
                per_platform = per_fn[platform_key]
                c = ','.join((str(count) for count in per_platform['counts']))
                profile_f.write('%s %s %s\n' % (test_key, platform_key, c))
        profile_f.close()