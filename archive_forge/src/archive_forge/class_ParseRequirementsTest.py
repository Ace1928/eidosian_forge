import email
import email.errors
import os
import re
import sysconfig
import tempfile
import textwrap
import fixtures
import pkg_resources
import six
import testscenarios
import testtools
from testtools import matchers
import virtualenv
from wheel import wheelfile
from pbr import git
from pbr import packaging
from pbr.tests import base
class ParseRequirementsTest(base.BaseTestCase):

    def test_empty_requirements(self):
        actual = packaging.parse_requirements([])
        self.assertEqual([], actual)

    def test_default_requirements(self):
        """Ensure default files used if no files provided."""
        tempdir = tempfile.mkdtemp()
        requirements = os.path.join(tempdir, 'requirements.txt')
        with open(requirements, 'w') as f:
            f.write('pbr')
        with mock.patch.object(packaging, 'REQUIREMENTS_FILES', (requirements,)):
            result = packaging.parse_requirements()
        self.assertEqual(['pbr'], result)

    def test_override_with_env(self):
        """Ensure environment variable used if no files provided."""
        _, tmp_file = tempfile.mkstemp(prefix='openstack', suffix='.setup')
        with open(tmp_file, 'w') as fh:
            fh.write('foo\nbar')
        self.useFixture(fixtures.EnvironmentVariable('PBR_REQUIREMENTS_FILES', tmp_file))
        self.assertEqual(['foo', 'bar'], packaging.parse_requirements())

    def test_override_with_env_multiple_files(self):
        _, tmp_file = tempfile.mkstemp(prefix='openstack', suffix='.setup')
        with open(tmp_file, 'w') as fh:
            fh.write('foo\nbar')
        self.useFixture(fixtures.EnvironmentVariable('PBR_REQUIREMENTS_FILES', 'no-such-file,' + tmp_file))
        self.assertEqual(['foo', 'bar'], packaging.parse_requirements())

    def test_index_present(self):
        tempdir = tempfile.mkdtemp()
        requirements = os.path.join(tempdir, 'requirements.txt')
        with open(requirements, 'w') as f:
            f.write('-i https://myindex.local\n')
            f.write('  --index-url https://myindex.local\n')
            f.write(' --extra-index-url https://myindex.local\n')
            f.write('--find-links https://myindex.local\n')
            f.write('arequirement>=1.0\n')
        result = packaging.parse_requirements([requirements])
        self.assertEqual(['arequirement>=1.0'], result)

    def test_nested_requirements(self):
        tempdir = tempfile.mkdtemp()
        requirements = os.path.join(tempdir, 'requirements.txt')
        nested = os.path.join(tempdir, 'nested.txt')
        with open(requirements, 'w') as f:
            f.write('-r ' + nested)
        with open(nested, 'w') as f:
            f.write('pbr')
        result = packaging.parse_requirements([requirements])
        self.assertEqual(['pbr'], result)