import os
import time
import sys
from io import StringIO, BytesIO
from pyomo.common.log import LoggingIntercept
import pyomo.common.unittest as unittest
from pyomo.common.tempfiles import TempfileManager
import pyomo.common.tee as tee
class TestTeeStream(unittest.TestCase):

    def test_stdout(self):
        a = StringIO()
        b = StringIO()
        with tee.TeeStream(a, b) as t:
            t.STDOUT.write('Hello\n')
        self.assertEqual(a.getvalue(), 'Hello\n')
        self.assertEqual(b.getvalue(), 'Hello\n')

    def test_err_and_out_are_different(self):
        with tee.TeeStream() as t:
            out = t.STDOUT
            self.assertIs(out, t.STDOUT)
            err = t.STDERR
            self.assertIs(err, t.STDERR)
            self.assertIsNot(out, err)

    @unittest.skipIf(not tee._peek_available, 'Requires the _mergedReader, but _peek_available==False')
    def test_merge_out_and_err(self):
        a = StringIO()
        b = StringIO()
        assert tee._poll_interval <= 0.1
        with tee.TeeStream(a, b) as t:
            t.STDOUT.write('Hello\nWorld')
            t.STDOUT.flush()
            time.sleep(tee._poll_interval * 100)
            t.STDERR.write('interrupting\ncow')
            t.STDERR.flush()
            start_time = time.time()
            while 'cow' not in a.getvalue() and time.time() - start_time < 1:
                time.sleep(tee._poll_interval)
        acceptable_results = {'Hello\ninterrupting\ncowWorld', 'interrupting\ncowHello\nWorld'}
        self.assertIn(a.getvalue(), acceptable_results)
        self.assertEqual(b.getvalue(), a.getvalue())

    def test_merged_out_and_err_without_peek(self):
        a = StringIO()
        b = StringIO()
        try:
            _tmp, tee._peek_available = (tee._peek_available, False)
            with tee.TeeStream(a, b) as t:
                t.STDOUT
                t.STDERR
                t.STDERR.write('Hello\n')
                t.STDERR.flush()
                time.sleep(tee._poll_interval * 2)
                t.STDOUT.write('World\n')
        finally:
            tee._peek_available = _tmp
        self.assertEqual(a.getvalue(), 'Hello\nWorld\n')
        self.assertEqual(b.getvalue(), 'Hello\nWorld\n')

    def test_binary_tee(self):
        a = BytesIO()
        b = BytesIO()
        with tee.TeeStream(a, b) as t:
            t.open('wb').write(b'Hello\n')
        self.assertEqual(a.getvalue(), b'Hello\n')
        self.assertEqual(b.getvalue(), b'Hello\n')

    def test_decoder_and_buffer_errors(self):
        ref = 'Hello, ©'
        bytes_ref = ref.encode()
        log = StringIO()
        with LoggingIntercept(log):
            with tee.TeeStream(encoding='utf-8') as t:
                os.write(t.STDOUT.fileno(), bytes_ref[:-1])
        self.assertEqual(log.getvalue(), "Stream handle closed with a partial line in the output buffer that was not emitted to the output stream(s):\n\t'Hello, '\nStream handle closed with un-decoded characters in the decoder buffer that was not emitted to the output stream(s):\n\tb'\\xc2'\n")
        out = StringIO()
        log = StringIO()
        with LoggingIntercept(log):
            with tee.TeeStream(out) as t:
                out.close()
                t.STDOUT.write('hi\n')
        self.assertRegex(log.getvalue(), "^Output stream \\(<.*?>\\) closed before all output was written to it. The following was left in the output buffer:\\n\\t'hi\\\\n'\\n$")

    def test_capture_output(self):
        out = StringIO()
        with tee.capture_output(out) as OUT:
            print('Hello World')
        self.assertEqual(OUT.getvalue(), 'Hello World\n')

    def test_duplicate_capture_output(self):
        out = StringIO()
        capture = tee.capture_output(out)
        capture.setup()
        try:
            with self.assertRaisesRegex(RuntimeError, 'Duplicate call to capture_output.setup'):
                capture.setup()
        finally:
            capture.reset()

    def test_capture_output_logfile_string(self):
        with TempfileManager.new_context() as tempfile:
            logfile = tempfile.create_tempfile()
            self.assertTrue(isinstance(logfile, str))
            with tee.capture_output(logfile):
                print('HELLO WORLD')
            with open(logfile, 'r') as f:
                result = f.read()
            self.assertEqual('HELLO WORLD\n', result)

    def test_capture_output_stack_error(self):
        OUT1 = StringIO()
        OUT2 = StringIO()
        old = (sys.stdout, sys.stderr)
        try:
            a = tee.capture_output(OUT1)
            a.setup()
            b = tee.capture_output(OUT2)
            b.setup()
            with self.assertRaisesRegex(RuntimeError, 'Captured output does not match sys.stdout'):
                a.reset()
            b.tee = None
        finally:
            sys.stdout, sys.stderr = old

    def test_deadlock(self):

        class MockStream(object):

            def write(self, data):
                time.sleep(0.2)
        _save = (tee._poll_timeout, tee._poll_timeout_deadlock)
        tee._poll_timeout = tee._poll_interval * 2 ** 5
        tee._poll_timeout_deadlock = tee._poll_interval * 2 ** 7
        try:
            with LoggingIntercept() as LOG, self.assertRaisesRegex(RuntimeError, 'deadlock'):
                with tee.TeeStream(MockStream()) as t:
                    err = t.STDERR
                    err.write('*')
            self.assertEqual('Significant delay observed waiting to join reader threads, possible output stream deadlock\n', LOG.getvalue())
        finally:
            tee._poll_timeout, tee._poll_timeout_deadlock = _save