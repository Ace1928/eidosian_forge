import glob
import os
import sys
import subprocess
import tempfile
import shutil
import atexit
import textwrap
import re
import pytest
import contextlib
import numpy
from pathlib import Path
from numpy.compat import asstr
from numpy._utils import asunicode
from numpy.testing import temppath, IS_WASM
from importlib import import_module
import os
import sys
def _get_compiler_status():
    global _compiler_status
    if _compiler_status is not None:
        return _compiler_status
    _compiler_status = (False, False, False)
    if IS_WASM:
        return _compiler_status
    code = textwrap.dedent(f"        import os\n        import sys\n        sys.path = {repr(sys.path)}\n\n        def configuration(parent_name='',top_path=None):\n            global config\n            from numpy.distutils.misc_util import Configuration\n            config = Configuration('', parent_name, top_path)\n            return config\n\n        from numpy.distutils.core import setup\n        setup(configuration=configuration)\n\n        config_cmd = config.get_config_cmd()\n        have_c = config_cmd.try_compile('void foo() {{}}')\n        print('COMPILERS:%%d,%%d,%%d' %% (have_c,\n                                          config.have_f77c(),\n                                          config.have_f90c()))\n        sys.exit(99)\n        ")
    code = code % dict(syspath=repr(sys.path))
    tmpdir = tempfile.mkdtemp()
    try:
        script = os.path.join(tmpdir, 'setup.py')
        with open(script, 'w') as f:
            f.write(code)
        cmd = [sys.executable, 'setup.py', 'config']
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=tmpdir)
        out, err = p.communicate()
    finally:
        shutil.rmtree(tmpdir)
    m = re.search(b'COMPILERS:(\\d+),(\\d+),(\\d+)', out)
    if m:
        _compiler_status = (bool(int(m.group(1))), bool(int(m.group(2))), bool(int(m.group(3))))
    return _compiler_status