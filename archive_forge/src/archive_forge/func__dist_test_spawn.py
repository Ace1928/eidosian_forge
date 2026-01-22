import atexit
import inspect
import os
import pprint
import re
import subprocess
import textwrap
@staticmethod
def _dist_test_spawn(cmd, display=None):
    try:
        o = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
        if o and re.match(_Distutils._dist_warn_regex, o):
            _Distutils.dist_error('Flags in command', cmd, "aren't supported by the compiler, output -> \n%s" % o)
    except subprocess.CalledProcessError as exc:
        o = exc.output
        s = exc.returncode
    except OSError as e:
        o = e
        s = 127
    else:
        return None
    _Distutils.dist_error('Command', cmd, 'failed with exit status %d output -> \n%s' % (s, o))