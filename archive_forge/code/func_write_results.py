import io
import linecache
import os
import sys
import sysconfig
import token
import tokenize
import inspect
import gc
import dis
import pickle
from time import monotonic as _time
import threading
def write_results(self, show_missing=True, summary=False, coverdir=None):
    """
        Write the coverage results.

        :param show_missing: Show lines that had no hits.
        :param summary: Include coverage summary per module.
        :param coverdir: If None, the results of each module are placed in its
                         directory, otherwise it is included in the directory
                         specified.
        """
    if self.calledfuncs:
        print()
        print('functions called:')
        calls = self.calledfuncs
        for filename, modulename, funcname in sorted(calls):
            print('filename: %s, modulename: %s, funcname: %s' % (filename, modulename, funcname))
    if self.callers:
        print()
        print('calling relationships:')
        lastfile = lastcfile = ''
        for (pfile, pmod, pfunc), (cfile, cmod, cfunc) in sorted(self.callers):
            if pfile != lastfile:
                print()
                print('***', pfile, '***')
                lastfile = pfile
                lastcfile = ''
            if cfile != pfile and lastcfile != cfile:
                print('  -->', cfile)
                lastcfile = cfile
            print('    %s.%s -> %s.%s' % (pmod, pfunc, cmod, cfunc))
    per_file = {}
    for filename, lineno in self.counts:
        lines_hit = per_file[filename] = per_file.get(filename, {})
        lines_hit[lineno] = self.counts[filename, lineno]
    sums = {}
    for filename, count in per_file.items():
        if self.is_ignored_filename(filename):
            continue
        if filename.endswith('.pyc'):
            filename = filename[:-1]
        if coverdir is None:
            dir = os.path.dirname(os.path.abspath(filename))
            modulename = _modname(filename)
        else:
            dir = coverdir
            os.makedirs(dir, exist_ok=True)
            modulename = _fullmodname(filename)
        if show_missing:
            lnotab = _find_executable_linenos(filename)
        else:
            lnotab = {}
        source = linecache.getlines(filename)
        coverpath = os.path.join(dir, modulename + '.cover')
        with open(filename, 'rb') as fp:
            encoding, _ = tokenize.detect_encoding(fp.readline)
        n_hits, n_lines = self.write_results_file(coverpath, source, lnotab, count, encoding)
        if summary and n_lines:
            percent = int(100 * n_hits / n_lines)
            sums[modulename] = (n_lines, percent, modulename, filename)
    if summary and sums:
        print('lines   cov%   module   (path)')
        for m in sorted(sums):
            n_lines, percent, modulename, filename = sums[m]
            print('%5d   %3d%%   %s   (%s)' % sums[m])
    if self.outfile:
        try:
            with open(self.outfile, 'wb') as f:
                pickle.dump((self.counts, self.calledfuncs, self.callers), f, 1)
        except OSError as err:
            print("Can't save counts files because %s" % err, file=sys.stderr)