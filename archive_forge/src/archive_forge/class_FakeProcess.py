import random
import subprocess
import sys
from fixtures import Fixture
class FakeProcess(object):
    """A test double process, roughly meeting subprocess.Popen's contract."""

    def __init__(self, args, info):
        self._args = args
        self.stdin = info.get('stdin')
        self.stdout = info.get('stdout')
        self.stderr = info.get('stderr')
        self.pid = random.randint(0, 65536)
        self._returncode = info.get('returncode', 0)
        self.returncode = None

    @property
    def args(self):
        return self._args['args']

    def poll(self):
        """Get the current value of FakeProcess.returncode.

        The returncode is None before communicate() and/or wait() are called,
        and it's set to the value provided by the 'info' dictionary otherwise
        (or 0 in case 'info' doesn't specify a value).
        """
        return self.returncode

    def communicate(self, input=None, timeout=None):
        self.returncode = self._returncode
        if self.stdin and input:
            self.stdin.write(input)
        if self.stdout:
            out = self.stdout.getvalue()
        else:
            out = ''
        if self.stderr:
            err = self.stderr.getvalue()
        else:
            err = ''
        return (out, err)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.wait()

    def kill(self):
        pass

    def wait(self, timeout=None, endtime=None):
        if self.returncode is None:
            self.communicate()
        return self.returncode