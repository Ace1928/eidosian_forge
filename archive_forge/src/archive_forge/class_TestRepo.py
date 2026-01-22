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
class TestRepo(fixtures.Fixture):
    """A git repo for testing with.

    Use of TempHomeDir with this fixture is strongly recommended as due to the
    lack of config --local in older gits, it will write to the users global
    configuration without TempHomeDir.
    """

    def __init__(self, basedir):
        super(TestRepo, self).__init__()
        self._basedir = basedir

    def setUp(self):
        super(TestRepo, self).setUp()
        base._run_cmd(['git', 'init', '.'], self._basedir)
        base._config_git()
        base._run_cmd(['git', 'add', '.'], self._basedir)

    def commit(self, message_content='test commit'):
        files = len(os.listdir(self._basedir))
        path = self._basedir + '/%d' % files
        open(path, 'wt').close()
        base._run_cmd(['git', 'add', path], self._basedir)
        base._run_cmd(['git', 'commit', '-m', message_content], self._basedir)

    def uncommit(self):
        base._run_cmd(['git', 'reset', '--hard', 'HEAD^'], self._basedir)

    def tag(self, version):
        base._run_cmd(['git', 'tag', '-sm', 'test tag', version], self._basedir)