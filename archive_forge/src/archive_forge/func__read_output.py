import os
import re
import sys
def _read_output(commandstring, capture_stderr=False):
    """Output from successful command execution or None"""
    import contextlib
    try:
        import tempfile
        fp = tempfile.NamedTemporaryFile()
    except ImportError:
        fp = open('/tmp/_osx_support.%s' % (os.getpid(),), 'w+b')
    with contextlib.closing(fp) as fp:
        if capture_stderr:
            cmd = "%s >'%s' 2>&1" % (commandstring, fp.name)
        else:
            cmd = "%s 2>/dev/null >'%s'" % (commandstring, fp.name)
        return fp.read().decode('utf-8').strip() if not os.system(cmd) else None