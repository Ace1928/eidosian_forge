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
def test_preversion_too_low_semver_headers(self):
    self.repo.commit()
    self.repo.tag('1.2.3')
    self.repo.commit('sem-ver: feature')
    err = self.assertRaises(ValueError, packaging._get_version_from_git, '1.2.4')
    self.assertThat(err.args[0], matchers.StartsWith('git history'))