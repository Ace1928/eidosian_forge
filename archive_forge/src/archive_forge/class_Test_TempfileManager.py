import gc
import glob
import os
import shutil
import sys
import tempfile
from io import StringIO
from os.path import abspath, dirname
import pyomo.common.unittest as unittest
import pyomo.common.tempfiles as tempfiles
from pyomo.common.dependencies import pyutilib_available
from pyomo.common.log import LoggingIntercept
from pyomo.common.tempfiles import (
class Test_TempfileManager(unittest.TestCase):

    def setUp(self):
        self.TM = TempfileManagerClass()

    def tearDown(self):
        self.TM.shutdown()

    def test_create_tempfile(self):
        context = self.TM.push()
        fname = self.TM.create_tempfile('suffix', 'prefix')
        self.assertRegex(os.path.basename(fname), '^prefix')
        self.assertRegex(os.path.basename(fname), 'suffix$')
        self.assertGreater(len(os.path.basename(fname)), len('prefixsuffix'))
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.isfile(fname))
        context.release()
        self.assertFalse(os.path.exists(fname))

    def test_mkstemp(self):
        context = self.TM.new_context()
        fd, fname = context.mkstemp('suffix', 'prefix')
        self.assertRegex(os.path.basename(fname), '^prefix')
        self.assertRegex(os.path.basename(fname), 'suffix$')
        self.assertGreater(len(os.path.basename(fname)), len('prefixsuffix'))
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.isfile(fname))
        os.fsync(fd)
        context.release()
        self.assertFalse(os.path.exists(fname))
        with self.assertRaises(OSError):
            os.fsync(fd)
        context = self.TM.new_context()
        fd, fname = context.mkstemp('suffix', 'prefix')
        os.close(fd)
        context.release()

    def test_create_tempdir(self):
        context = self.TM.push()
        fname = self.TM.create_tempdir('suffix', 'prefix')
        self.assertRegex(os.path.basename(fname), '^prefix')
        self.assertRegex(os.path.basename(fname), 'suffix$')
        self.assertGreater(len(os.path.basename(fname)), len('prefixsuffix'))
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.isdir(fname))
        context.release()
        self.assertFalse(os.path.exists(fname))

    def test_add_tempfile(self):
        context1 = self.TM.push()
        context2 = self.TM.push()
        fname = context1.create_tempfile()
        dname = context1.create_tempdir()
        sub_fname = os.path.join(dname, 'testfile')
        self.TM.add_tempfile(fname)
        with self.assertRaisesRegex(IOError, 'Temporary file does not exist: %s' % sub_fname.replace('\\', '\\\\')):
            self.TM.add_tempfile(sub_fname)
        self.TM.add_tempfile(sub_fname, exists=False)
        with open(sub_fname, 'w') as FILE:
            FILE.write('\n')
        self.assertTrue(os.path.exists(fname))
        self.assertTrue(os.path.exists(dname))
        self.assertTrue(os.path.exists(sub_fname))
        self.TM.pop()
        self.assertFalse(os.path.exists(fname))
        self.assertTrue(os.path.exists(dname))
        self.assertFalse(os.path.exists(sub_fname))
        self.TM.pop()
        self.assertFalse(os.path.exists(fname))
        self.assertFalse(os.path.exists(dname))
        self.assertFalse(os.path.exists(sub_fname))

    def test_sequential_files(self):
        with LoggingIntercept() as LOG:
            self.assertIsNone(self.TM.sequential_files())
        self.assertIn('The TempfileManager.sequential_files() method has been removed', LOG.getvalue().replace('\n', ' '))
        self.assertIsNone(self.TM.unique_files())

    def test_gettempprefix(self):
        ctx = self.TM.new_context()
        pre = ctx.gettempprefix()
        self.assertIsInstance(pre, str)
        self.assertEqual(pre, tempfile.gettempprefix())
        preb = ctx.gettempprefixb()
        self.assertIsInstance(preb, bytes)
        self.assertEqual(preb, tempfile.gettempprefixb())

    def test_gettempdir(self):
        context = self.TM.push()
        fname = context.create_tempfile()
        self.assertIsInstance(fname, str)
        system_tmpdir = os.path.dirname(fname)
        self.assertEqual(system_tmpdir, tempfile.gettempdir())
        tmpdir = context.gettempdir()
        self.assertIsInstance(tmpdir, str)
        self.assertEqual(tmpdir, system_tmpdir)
        tmpdirb = context.gettempdirb()
        self.assertIsInstance(tmpdirb, bytes)
        self.assertEqual(tmpdirb.decode(), tmpdir)
        manager_tmpdir = context.create_tempdir()
        self.assertNotEqual(manager_tmpdir, system_tmpdir)
        self.TM.tempdir = manager_tmpdir
        fname = context.create_tempfile()
        self.assertIsInstance(fname, str)
        tmpdir = context.gettempdir()
        self.assertIsInstance(tmpdir, str)
        self.assertEqual(tmpdir, manager_tmpdir)
        tmpdirb = context.gettempdirb()
        self.assertIsInstance(tmpdirb, bytes)
        self.assertEqual(tmpdirb.decode(), tmpdir)
        context_tmpdir = context.create_tempdir()
        self.assertNotEqual(context_tmpdir, system_tmpdir)
        self.assertNotEqual(context_tmpdir, manager_tmpdir)
        context.tempdir = context_tmpdir
        fname = context.create_tempfile()
        self.assertIsInstance(fname, str)
        tmpdir = context.gettempdir()
        self.assertIsInstance(tmpdir, str)
        self.assertEqual(tmpdir, context_tmpdir)
        tmpdirb = context.gettempdirb()
        self.assertIsInstance(tmpdirb, bytes)
        self.assertEqual(tmpdirb.decode(), tmpdir)
        context.tempdir = context_tmpdir.encode()
        fname = context.create_tempfile()
        self.assertIsInstance(fname, bytes)
        tmpdir = context.gettempdir()
        self.assertIsInstance(tmpdir, str)
        self.assertEqual(tmpdir, context_tmpdir)
        tmpdirb = context.gettempdirb()
        self.assertIsInstance(tmpdirb, bytes)
        self.assertEqual(tmpdirb.decode(), tmpdir)
        self.TM.pop()

    def test_shutdown(self):
        with LoggingIntercept() as LOG:
            self.TM.shutdown()
        self.assertEqual(LOG.getvalue(), '')
        self.TM = TempfileManagerClass()
        ctx = self.TM.push()
        with LoggingIntercept() as LOG:
            self.TM.shutdown()
        self.assertEqual(LOG.getvalue().strip(), 'TempfileManagerClass instance: un-popped tempfile contexts still exist during TempfileManager instance shutdown')
        self.TM = TempfileManagerClass()
        ctx = self.TM.push()
        fname = ctx.create_tempfile()
        self.assertTrue(os.path.exists(fname))
        with LoggingIntercept() as LOG:
            self.TM.shutdown()
        self.assertFalse(os.path.exists(fname))
        self.assertEqual(LOG.getvalue().strip(), 'Temporary files created through TempfileManager contexts have not been deleted (observed during TempfileManager instance shutdown).\nUndeleted entries:\n\t%s\nTempfileManagerClass instance: un-popped tempfile contexts still exist during TempfileManager instance shutdown' % fname)
        with LoggingIntercept() as LOG:
            self.TM.shutdown()
        self.assertEqual(LOG.getvalue(), '')

    def test_del_clears_contexts(self):
        TM = TempfileManagerClass()
        ctx = TM.push()
        fname = ctx.create_tempfile()
        self.assertTrue(os.path.exists(fname))
        with LoggingIntercept() as LOG:
            TM = None
            gc.collect()
            gc.collect()
            gc.collect()
        self.assertFalse(os.path.exists(fname))
        self.assertEqual(LOG.getvalue().strip(), 'Temporary files created through TempfileManager contexts have not been deleted (observed during TempfileManager instance shutdown).\nUndeleted entries:\n\t%s\nTempfileManagerClass instance: un-popped tempfile contexts still exist during TempfileManager instance shutdown' % fname)

    def test_tempfilemanager_as_context_manager(self):
        with LoggingIntercept() as LOG:
            with self.TM:
                fname = self.TM.create_tempfile()
                self.assertTrue(os.path.exists(fname))
            self.assertFalse(os.path.exists(fname))
            self.assertEqual(LOG.getvalue(), '')
            with self.TM:
                self.TM.push()
                fname = self.TM.create_tempfile()
                self.assertTrue(os.path.exists(fname))
            self.assertFalse(os.path.exists(fname))
            self.assertEqual(LOG.getvalue().strip(), 'TempfileManager: tempfile context was pushed onto the TempfileManager stack within a context manager (i.e., `with TempfileManager:`) but was not popped before the context manager exited.  Popping the context to preserve the stack integrity.')

    def test_tempfilecontext_as_context_manager(self):
        with LoggingIntercept() as LOG:
            ctx = self.TM.new_context()
            with ctx:
                fname = ctx.create_tempfile()
                self.assertTrue(os.path.exists(fname))
            self.assertFalse(os.path.exists(fname))
            self.assertEqual(LOG.getvalue(), '')

    @unittest.skipIf(not sys.platform.lower().startswith('win'), 'test only applies to Windows platforms')
    def test_open_tempfile_windows(self):
        self.TM.push()
        fname = self.TM.create_tempfile()
        f = open(fname)
        try:
            _orig = tempfiles.deletion_errors_are_fatal
            tempfiles.deletion_errors_are_fatal = True
            with self.assertRaisesRegex(WindowsError, '.*process cannot access the file'):
                self.TM.pop()
        finally:
            tempfiles.deletion_errors_are_fatal = _orig
            f.close()
            os.remove(fname)
        self.TM.push()
        fname = self.TM.create_tempfile()
        f = open(fname)
        try:
            _orig = tempfiles.deletion_errors_are_fatal
            tempfiles.deletion_errors_are_fatal = False
            with LoggingIntercept(None, 'pyomo.common') as LOG:
                self.TM.pop()
            self.assertIn('Unable to delete temporary file', LOG.getvalue())
        finally:
            tempfiles.deletion_errors_are_fatal = _orig
            f.close()
            os.remove(fname)

    @unittest.skipUnless(pyutilib_available, 'deprecation test requires pyutilib')
    def test_deprecated_tempdir(self):
        self.TM.push()
        try:
            tmpdir = self.TM.create_tempdir()
            _orig = tempfiles.pyutilib_tempfiles.TempfileManager.tempdir
            tempfiles.pyutilib_tempfiles.TempfileManager.tempdir = tmpdir
            self.TM.tempdir = None
            with LoggingIntercept() as LOG:
                fname = self.TM.create_tempfile()
            self.assertIn('The use of the PyUtilib TempfileManager.tempdir to specify the default location for Pyomo temporary files', LOG.getvalue().replace('\n', ' '))
            with LoggingIntercept() as LOG:
                dname = self.TM.create_tempdir()
            self.assertIn('The use of the PyUtilib TempfileManager.tempdir to specify the default location for Pyomo temporary files', LOG.getvalue().replace('\n', ' '))
        finally:
            self.TM.pop()
            tempfiles.pyutilib_tempfiles.TempfileManager.tempdir = _orig

    def test_context(self):
        with self.assertRaisesRegex(TempfileContextError, 'TempfileManager has no currently active context'):
            self.TM.context()
        ctx = self.TM.push()
        self.assertIs(ctx, self.TM.context())