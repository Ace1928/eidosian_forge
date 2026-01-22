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
class TestPEP517Support(base.BaseTestCase):

    def test_pep_517_support(self):
        pkgs = {'test_pep517': {'requirements.txt': textwrap.dedent('                        sphinx\n                        iso8601\n                    '), 'setup.py': textwrap.dedent('                        #!/usr/bin/env python\n                        import setuptools\n                        setuptools.setup(pbr=True)\n                    '), 'setup.cfg': textwrap.dedent('                        [metadata]\n                        name = test_pep517\n                        summary = A tiny test project\n                        author = PBR Team\n                        author_email = foo@example.com\n                        home_page = https://example.com/\n                        classifier =\n                            Intended Audience :: Information Technology\n                            Intended Audience :: System Administrators\n                            License :: OSI Approved :: Apache Software License\n                            Operating System :: POSIX :: Linux\n                            Programming Language :: Python\n                            Programming Language :: Python :: 2\n                            Programming Language :: Python :: 2.7\n                            Programming Language :: Python :: 3\n                            Programming Language :: Python :: 3.6\n                            Programming Language :: Python :: 3.7\n                            Programming Language :: Python :: 3.8\n                    '), 'pyproject.toml': textwrap.dedent('                        [build-system]\n                        requires = ["pbr", "setuptools>=36.6.0", "wheel"]\n                        build-backend = "pbr.build"\n                    ')}}
        pkg_dirs = self.useFixture(CreatePackages(pkgs)).package_dirs
        pkg_dir = pkg_dirs['test_pep517']
        venv = self.useFixture(Venv('PEP517'))
        self._run_cmd(venv.python, ('-m', 'build', '--no-isolation', '.'), allow_fail=False, cwd=pkg_dir)