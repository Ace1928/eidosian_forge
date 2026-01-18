import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
@testtools.skipUnless(sys.version_info < (3, 10), 'only relevant on Python <3.10')
def test_rejects_3_10_args_on_older_versions(self):
    fixture = self.useFixture(FakePopen(lambda proc_args: {}))
    with testtools.ExpectedException(TypeError, ".* got an unexpected keyword argument 'pipesize'"):
        fixture(args='args', pipesize=1024)