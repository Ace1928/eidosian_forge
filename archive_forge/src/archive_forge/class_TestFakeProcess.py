import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
class TestFakeProcess(testtools.TestCase):

    def test_wait(self):
        proc = FakeProcess({}, {})
        proc.returncode = 45
        self.assertEqual(45, proc.wait())

    def test_communicate(self):
        proc = FakeProcess({}, {})
        self.assertEqual(('', ''), proc.communicate())
        self.assertEqual(0, proc.returncode)

    def test_communicate_with_out(self):
        proc = FakeProcess({}, {'stdout': io.BytesIO(b'foo')})
        self.assertEqual((b'foo', ''), proc.communicate())
        self.assertEqual(0, proc.returncode)

    def test_communicate_with_input(self):
        proc = FakeProcess({}, {'stdout': io.BytesIO(b'foo')})
        self.assertEqual((b'foo', ''), proc.communicate(input=b'bar'))

    def test_communicate_with_input_and_stdin(self):
        stdin = io.BytesIO()
        proc = FakeProcess({}, {'stdin': stdin})
        proc.communicate(input=b'hello')
        self.assertEqual(b'hello', stdin.getvalue())

    def test_communicate_with_timeout(self):
        proc = FakeProcess({}, {'stdout': io.BytesIO(b'foo')})
        self.assertEqual((b'foo', ''), proc.communicate(timeout=10))

    def test_args(self):
        proc = FakeProcess({'args': ['ls', '-lh']}, {})
        proc.returncode = 45
        self.assertEqual(45, proc.wait())
        self.assertEqual(proc.args, ['ls', '-lh'])

    def test_kill(self):
        proc = FakeProcess({}, {})
        self.assertIs(None, proc.kill())

    def test_poll(self):
        proc = FakeProcess({}, {})
        self.assertIs(None, proc.poll())
        proc.communicate()
        self.assertEqual(0, proc.poll())

    def test_poll_with_returncode(self):
        proc = FakeProcess({}, {})
        proc.communicate()
        self.assertEqual(0, proc.poll())

    def test_wait_with_timeout_and_endtime(self):
        proc = FakeProcess({}, {})
        self.assertEqual(0, proc.wait(timeout=4, endtime=7))