import os
import re
import sys
import shutil
import warnings
import textwrap
import unittest
import tempfile
import subprocess
from distutils import ccompiler
import runtests
import Cython.Distutils.extension
import Cython.Distutils.old_build_ext as build_ext
from Cython.Debugger import Cygdb as cygdb
class DebuggerTestCase(unittest.TestCase):

    def setUp(self):
        """
        Run gdb and have cygdb import the debug information from the code
        defined in TestParseTreeTransforms's setUp method
        """
        if not test_gdb():
            return
        self.tempdir = tempfile.mkdtemp()
        self.destfile = os.path.join(self.tempdir, 'codefile.pyx')
        self.debug_dest = os.path.join(self.tempdir, 'cython_debug', 'cython_debug_info_codefile')
        self.cfuncs_destfile = os.path.join(self.tempdir, 'cfuncs')
        self.cwd = os.getcwd()
        try:
            os.chdir(self.tempdir)
            shutil.copy(codefile, self.destfile)
            shutil.copy(cfuncs_file, self.cfuncs_destfile + '.c')
            shutil.copy(cfuncs_file.replace('.c', '.h'), self.cfuncs_destfile + '.h')
            compiler = ccompiler.new_compiler()
            compiler.compile(['cfuncs.c'], debug=True, extra_postargs=['-fPIC'])
            opts = dict(test_directory=self.tempdir, module='codefile', module_path=self.destfile)
            optimization_disabler = build_ext.Optimization()
            cython_compile_testcase = runtests.CythonCompileTestCase(workdir=self.tempdir, cleanup_workdir=False, tags=runtests.parse_tags(codefile), **opts)
            new_stderr = open(os.devnull, 'w')
            stderr = sys.stderr
            sys.stderr = new_stderr
            optimization_disabler.disable_optimization()
            try:
                cython_compile_testcase.run_cython(targetdir=self.tempdir, incdir=None, annotate=False, extra_compile_options={'gdb_debug': True, 'output_dir': self.tempdir}, **opts)
                cython_compile_testcase.run_distutils(test_directory=opts['test_directory'], module=opts['module'], workdir=opts['test_directory'], incdir=None, extra_extension_args={'extra_objects': ['cfuncs.o']})
            finally:
                optimization_disabler.restore_state()
                sys.stderr = stderr
                new_stderr.close()
        except:
            os.chdir(self.cwd)
            raise

    def tearDown(self):
        if not test_gdb():
            return
        os.chdir(self.cwd)
        shutil.rmtree(self.tempdir)