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
def test_parse_dependency_normal(self):
    with open(self.tmp_file, 'w') as fh:
        fh.write('http://test.com\n')
    self.assertEqual(['http://test.com'], packaging.parse_dependency_links([self.tmp_file]))