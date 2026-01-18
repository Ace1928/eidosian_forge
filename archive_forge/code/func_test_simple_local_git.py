import os
from testtools import content
from .. import plugins as _mod_plugins
from .. import trace
from ..bzr.smart import medium
from ..controldir import ControlDir
from ..transport import remote
from . import TestCaseWithTransport
def test_simple_local_git(self):
    self.make_branch_and_tree('.', format='git')
    self.run_command_check_imports(['st'], ['breezy.annotate', 'breezy.bugtracker', 'breezy.bundle.commands', 'breezy.cmd_version_info', 'breezy.externalcommand', 'breezy.filters', 'breezy.gpg', 'breezy.info', 'breezy.merge', 'breezy.merge_directive', 'breezy.msgeditor', 'breezy.rules', 'breezy.sign_my_commits', 'breezy.bzr.hashcache', 'breezy.bzr.knit', 'breezy.bzr.remote', 'breezy.bzr.smart', 'breezy.bzr.smart.client', 'breezy.bzr.smart.medium', 'breezy.bzr.smart.server', 'breezy.transform', 'breezy.version_info_formats.format_rio', 'breezy.bzr.xml_serializer', 'breezy.bzr.xml8', 'breezy.bzr.inventory', 'breezy.bzr.bzrdir', 'breezy.git.remote', 'breezy.git.commit', 'getpass', 'kerberos', 'merge3', 'shutil', 'smtplib', 'ssl', 'tempfile', 'tarfile', 'termios', 'tty'] + old_format_modules)