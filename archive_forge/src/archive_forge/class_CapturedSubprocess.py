import os
import shutil
import subprocess
import sys
import fixtures
import testresources
import testtools
from testtools import content
from pbr import options
class CapturedSubprocess(fixtures.Fixture):
    """Run a process and capture its output.

    :attr stdout: The output (a string).
    :attr stderr: The standard error (a string).
    :attr returncode: The return code of the process.

    Note that stdout and stderr are decoded from the bytestrings subprocess
    returns using error=replace.
    """

    def __init__(self, label, *args, **kwargs):
        """Create a CapturedSubprocess.

        :param label: A label for the subprocess in the test log. E.g. 'foo'.
        :param *args: The *args to pass to Popen.
        :param **kwargs: The **kwargs to pass to Popen.
        """
        super(CapturedSubprocess, self).__init__()
        self.label = label
        self.args = args
        self.kwargs = kwargs
        self.kwargs['stderr'] = subprocess.PIPE
        self.kwargs['stdin'] = subprocess.PIPE
        self.kwargs['stdout'] = subprocess.PIPE

    def setUp(self):
        super(CapturedSubprocess, self).setUp()
        proc = subprocess.Popen(*self.args, **self.kwargs)
        out, err = proc.communicate()
        self.out = out.decode('utf-8', 'replace')
        self.err = err.decode('utf-8', 'replace')
        self.addDetail(self.label + '-stdout', content.text_content(self.out))
        self.addDetail(self.label + '-stderr', content.text_content(self.err))
        self.returncode = proc.returncode
        if proc.returncode:
            raise AssertionError('Failed process args=%r, kwargs=%r, returncode=%s' % (self.args, self.kwargs, proc.returncode))
        self.addCleanup(delattr, self, 'out')
        self.addCleanup(delattr, self, 'err')
        self.addCleanup(delattr, self, 'returncode')