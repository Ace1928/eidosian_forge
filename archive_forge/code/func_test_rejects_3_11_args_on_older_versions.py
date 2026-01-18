import inspect
import io
import subprocess
import sys
import testtools
from fixtures import FakePopen, TestWithFixtures
from fixtures._fixtures.popen import FakeProcess
@testtools.skipUnless(sys.version_info < (3, 11), 'only relevant on Python <3.11')
def test_rejects_3_11_args_on_older_versions(self):
    fixture = self.useFixture(FakePopen(lambda proc_args: {}))
    with testtools.ExpectedException(TypeError, ".* got an unexpected keyword argument 'process_group'"):
        fixture(args='args', process_group=42)