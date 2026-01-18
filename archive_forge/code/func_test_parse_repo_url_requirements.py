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
def test_parse_repo_url_requirements(self):
    result = packaging.parse_requirements([self.requirements])
    self.assertEqual(['oslo.messaging>=1.0.0-rc', 'django-thumborize', 'django-thumborize-beta', 'django-thumborize2-beta', 'django-thumborize2-beta>=4.0.1', 'django-thumborize2-beta>=1.0.0-alpha.beta.1', 'django-thumborize2-beta>=1.0.0-alpha-a.b-c-somethinglong+build.1-aef.1-its-okay', 'django-thumborize2-beta>=2.0.0-rc.1+build.123', 'Proj1', 'Proj2>=0.0.1', 'Proj3', 'Proj4>=0.0.2', 'Proj5', 'Proj>=0.0.3', 'Proj', 'Proj>=0.0.4', 'Proj', 'foo-bar>=1.2.4', 'pypi-proj1', 'pypi-proj2'], result)