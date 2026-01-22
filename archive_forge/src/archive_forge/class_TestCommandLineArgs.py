import io
import os
import sys
import subprocess
from test import support
import unittest
import unittest.test
from unittest.test.test_result import BufferedWriter
@support.requires_subprocess()
class TestCommandLineArgs(unittest.TestCase):

    def setUp(self):
        self.program = InitialisableProgram()
        self.program.createTests = lambda: None
        FakeRunner.initArgs = None
        FakeRunner.test = None
        FakeRunner.raiseError = 0

    def testVerbosity(self):
        program = self.program
        for opt in ('-q', '--quiet'):
            program.verbosity = 1
            program.parseArgs([None, opt])
            self.assertEqual(program.verbosity, 0)
        for opt in ('-v', '--verbose'):
            program.verbosity = 1
            program.parseArgs([None, opt])
            self.assertEqual(program.verbosity, 2)

    def testBufferCatchFailfast(self):
        program = self.program
        for arg, attr in (('buffer', 'buffer'), ('failfast', 'failfast'), ('catch', 'catchbreak')):
            setattr(program, attr, None)
            program.parseArgs([None])
            self.assertIs(getattr(program, attr), False)
            false = []
            setattr(program, attr, false)
            program.parseArgs([None])
            self.assertIs(getattr(program, attr), false)
            true = [42]
            setattr(program, attr, true)
            program.parseArgs([None])
            self.assertIs(getattr(program, attr), true)
            short_opt = '-%s' % arg[0]
            long_opt = '--%s' % arg
            for opt in (short_opt, long_opt):
                setattr(program, attr, None)
                program.parseArgs([None, opt])
                self.assertIs(getattr(program, attr), True)
                setattr(program, attr, False)
                with support.captured_stderr() as stderr, self.assertRaises(SystemExit) as cm:
                    program.parseArgs([None, opt])
                self.assertEqual(cm.exception.args, (2,))
                setattr(program, attr, True)
                with support.captured_stderr() as stderr, self.assertRaises(SystemExit) as cm:
                    program.parseArgs([None, opt])
                self.assertEqual(cm.exception.args, (2,))

    def testWarning(self):
        """Test the warnings argument"""

        class FakeTP(unittest.TestProgram):

            def parseArgs(self, *args, **kw):
                pass

            def runTests(self, *args, **kw):
                pass
        warnoptions = sys.warnoptions[:]
        try:
            sys.warnoptions[:] = []
            self.assertEqual(FakeTP().warnings, 'default')
            self.assertEqual(FakeTP(warnings='ignore').warnings, 'ignore')
            sys.warnoptions[:] = ['somevalue']
            self.assertEqual(FakeTP().warnings, None)
            self.assertEqual(FakeTP(warnings='ignore').warnings, 'ignore')
        finally:
            sys.warnoptions[:] = warnoptions

    def testRunTestsRunnerClass(self):
        program = self.program
        program.testRunner = FakeRunner
        program.verbosity = 'verbosity'
        program.failfast = 'failfast'
        program.buffer = 'buffer'
        program.warnings = 'warnings'
        program.runTests()
        self.assertEqual(FakeRunner.initArgs, {'verbosity': 'verbosity', 'failfast': 'failfast', 'buffer': 'buffer', 'tb_locals': False, 'warnings': 'warnings'})
        self.assertEqual(FakeRunner.test, 'test')
        self.assertIs(program.result, RESULT)

    def testRunTestsRunnerInstance(self):
        program = self.program
        program.testRunner = FakeRunner()
        FakeRunner.initArgs = None
        program.runTests()
        self.assertIsNone(FakeRunner.initArgs)
        self.assertEqual(FakeRunner.test, 'test')
        self.assertIs(program.result, RESULT)

    def test_locals(self):
        program = self.program
        program.testRunner = FakeRunner
        program.parseArgs([None, '--locals'])
        self.assertEqual(True, program.tb_locals)
        program.runTests()
        self.assertEqual(FakeRunner.initArgs, {'buffer': False, 'failfast': False, 'tb_locals': True, 'verbosity': 1, 'warnings': None})

    def testRunTestsOldRunnerClass(self):
        program = self.program
        FakeRunner.raiseError = 2
        program.testRunner = FakeRunner
        program.verbosity = 'verbosity'
        program.failfast = 'failfast'
        program.buffer = 'buffer'
        program.test = 'test'
        program.runTests()
        self.assertEqual(FakeRunner.initArgs, {})
        self.assertEqual(FakeRunner.test, 'test')
        self.assertIs(program.result, RESULT)

    def testCatchBreakInstallsHandler(self):
        module = sys.modules['unittest.main']
        original = module.installHandler

        def restore():
            module.installHandler = original
        self.addCleanup(restore)
        self.installed = False

        def fakeInstallHandler():
            self.installed = True
        module.installHandler = fakeInstallHandler
        program = self.program
        program.catchbreak = True
        program.testRunner = FakeRunner
        program.runTests()
        self.assertTrue(self.installed)

    def _patch_isfile(self, names, exists=True):

        def isfile(path):
            return path in names
        original = os.path.isfile
        os.path.isfile = isfile

        def restore():
            os.path.isfile = original
        self.addCleanup(restore)

    def testParseArgsFileNames(self):
        program = self.program
        argv = ['progname', 'foo.py', 'bar.Py', 'baz.PY', 'wing.txt']
        self._patch_isfile(argv)
        program.createTests = lambda: None
        program.parseArgs(argv)
        expected = ['foo', 'bar', 'baz', 'wing.txt']
        self.assertEqual(program.testNames, expected)

    def testParseArgsFilePaths(self):
        program = self.program
        argv = ['progname', 'foo/bar/baz.py', 'green\\red.py']
        self._patch_isfile(argv)
        program.createTests = lambda: None
        program.parseArgs(argv)
        expected = ['foo.bar.baz', 'green.red']
        self.assertEqual(program.testNames, expected)

    def testParseArgsNonExistentFiles(self):
        program = self.program
        argv = ['progname', 'foo/bar/baz.py', 'green\\red.py']
        self._patch_isfile([])
        program.createTests = lambda: None
        program.parseArgs(argv)
        self.assertEqual(program.testNames, argv[1:])

    def testParseArgsAbsolutePathsThatCanBeConverted(self):
        cur_dir = os.getcwd()
        program = self.program

        def _join(name):
            return os.path.join(cur_dir, name)
        argv = ['progname', _join('foo/bar/baz.py'), _join('green\\red.py')]
        self._patch_isfile(argv)
        program.createTests = lambda: None
        program.parseArgs(argv)
        expected = ['foo.bar.baz', 'green.red']
        self.assertEqual(program.testNames, expected)

    def testParseArgsAbsolutePathsThatCannotBeConverted(self):
        program = self.program
        argv = ['progname', '/foo/bar/baz.py', '/green/red.py']
        self._patch_isfile(argv)
        program.createTests = lambda: None
        program.parseArgs(argv)
        self.assertEqual(program.testNames, argv[1:])

    def testParseArgsSelectedTestNames(self):
        program = self.program
        argv = ['progname', '-k', 'foo', '-k', 'bar', '-k', '*pat*']
        program.createTests = lambda: None
        program.parseArgs(argv)
        self.assertEqual(program.testNamePatterns, ['*foo*', '*bar*', '*pat*'])

    def testSelectedTestNamesFunctionalTest(self):

        def run_unittest(args):
            cmd = [sys.executable, '-E', '-m', 'unittest'] + args
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, cwd=os.path.dirname(__file__))
            with p:
                _, stderr = p.communicate()
            return stderr.decode()
        t = '_test_warnings'
        self.assertIn('Ran 7 tests', run_unittest([t]))
        self.assertIn('Ran 7 tests', run_unittest(['-k', 'TestWarnings', t]))
        self.assertIn('Ran 7 tests', run_unittest(['discover', '-p', '*_test*', '-k', 'TestWarnings']))
        self.assertIn('Ran 2 tests', run_unittest(['-k', 'f', t]))
        self.assertIn('Ran 7 tests', run_unittest(['-k', 't', t]))
        self.assertIn('Ran 3 tests', run_unittest(['-k', '*t', t]))
        self.assertIn('Ran 7 tests', run_unittest(['-k', '*test_warnings.*Warning*', t]))
        self.assertIn('Ran 1 test', run_unittest(['-k', '*test_warnings.*warning*', t]))