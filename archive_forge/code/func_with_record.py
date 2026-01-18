from __future__ import absolute_import, print_function
import cython
from .. import __version__
import collections
import contextlib
import hashlib
import os
import shutil
import subprocess
import re, sys, time
from glob import iglob
from io import open as io_open
from os.path import relpath as _relpath
import zipfile
from .. import Utils
from ..Utils import (cached_function, cached_method, path_exists,
from ..Compiler import Errors
from ..Compiler.Main import Context
from ..Compiler.Options import (CompilationOptions, default_options,
def with_record(*args):
    t = time.time()
    success = True
    try:
        try:
            func(*args)
        except:
            success = False
    finally:
        t = time.time() - t
        module = fully_qualified_name(args[0])
        name = 'cythonize.' + module
        failures = 1 - success
        if success:
            failure_item = ''
        else:
            failure_item = 'failure'
        output = open(os.path.join(compile_result_dir, name + '.xml'), 'w')
        output.write('\n                    <?xml version="1.0" ?>\n                    <testsuite name="%(name)s" errors="0" failures="%(failures)s" tests="1" time="%(t)s">\n                    <testcase classname="%(name)s" name="cythonize">\n                    %(failure_item)s\n                    </testcase>\n                    </testsuite>\n                '.strip() % locals())
        output.close()