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
def test_parse_dependency_with_git_egg_url(self):
    with open(self.tmp_file, 'w') as fh:
        fh.write('-e git://foo.com/zipball#egg=bar')
    self.assertEqual(['git://foo.com/zipball#egg=bar'], packaging.parse_dependency_links([self.tmp_file]))