import re
import sys
import subprocess
import pyomo.common.unittest as unittest
@unittest.skipIf('pypy_version_info' in dir(sys), "PyPy does not support '-X importtime")
def test_tpl_import_time(self):
    data = collect_import_time('pyomo.environ')
    pyomo_time = sum(data.pyomo.values())
    tpl_time = sum(data.tpl.values())
    total = float(pyomo_time + tpl_time)
    print('Pyomo (by module time):')
    print('\n'.join(('   %s: %s' % i for i in sorted(data.pyomo.items(), key=lambda x: x[1]))))
    print('TPLS:')
    _line_fmt = '   %%%ds: %%6d %%s' % (max((len(k[:k.find(' ')]) for k in data.tpl)),)
    print('\n'.join((_line_fmt % (k[:k.find(' ')], v, k[k.find(' '):]) for k, v in sorted(data.tpl.items()))))
    tpl = {}
    for k, v in data.tpl.items():
        _mod = k[:k.find(' ')].split('.')[0]
        tpl[_mod] = tpl.get(_mod, 0) + v
    tpl_by_time = sorted(tpl.items(), key=lambda x: x[1])
    print('TPLS (by package time):')
    print('\n'.join(('   %12s: %6d (%4.1f%%)' % (m, t, 100 * t / total) for m, t in tpl_by_time)))
    print('Pyomo:    %6d (%4.1f%%)' % (pyomo_time, 100 * pyomo_time / total))
    print('TPL:      %6d (%4.1f%%)' % (tpl_time, 100 * tpl_time / total))
    self.assertLess(tpl_time / total, 0.65)
    ref = {'__future__', 'argparse', 'ast', 'backports_abc', 'base64', 'cPickle', 'csv', 'ctypes', 'decimal', 'gc', 'glob', 'heapq', 'importlib', 'inspect', 'json', 'locale', 'logging', 'pickle', 'platform', 'shlex', 'socket', 'subprocess', 'tempfile', 'textwrap', 'typing', 'win32file', 'win32pipe'}
    ref.add('ply')
    diff = set((_[0] for _ in tpl_by_time[-5:])).difference(ref)
    self.assertEqual(diff, set(), 'Unexpected module found in 5 slowest-loading TPL modules')