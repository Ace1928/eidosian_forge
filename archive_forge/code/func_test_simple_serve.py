import os
from testtools import content
from .. import plugins as _mod_plugins
from .. import trace
from ..bzr.smart import medium
from ..controldir import ControlDir
from ..transport import remote
from . import TestCaseWithTransport
def test_simple_serve(self):
    tree = self.make_branch_and_tree('.')
    stderr_file = open('bzr-serve.stderr', 'w')
    process = self.start_brz_subprocess_with_import_check(['serve', '--inet', '-d', tree.basedir], stderr_file=stderr_file)
    url = 'bzr://localhost/'
    self.permit_url(url)
    client_medium = medium.SmartSimplePipesClientMedium(process.stdout, process.stdin, url)
    transport = remote.RemoteTransport(url, medium=client_medium)
    branch = ControlDir.open_from_transport(transport).open_branch()
    process.stdin.close()
    process.stdin = None
    out, err = self.finish_brz_subprocess(process, universal_newlines=False)
    stderr_file.close()
    with open('bzr-serve.stderr', 'rb') as stderr_file:
        err = stderr_file.read()
    self.check_forbidden_modules(err, ['breezy.annotate', 'breezy.atomicfile', 'breezy.bugtracker', 'breezy.bundle.commands', 'breezy.cmd_version_info', 'breezy.bzr.dirstate', 'breezy.bzr._dirstate_helpers_py', 'breezy.bzr._dirstate_helpers_pyx', 'breezy.externalcommand', 'breezy.filters', 'breezy.gpg', 'breezy.info', 'breezy.merge_directive', 'breezy.msgeditor', 'breezy.rules', 'breezy.sign_my_commits', 'breezy.transform', 'breezy.version_info_formats.format_rio', 'breezy.bzr.hashcache', 'breezy.bzr.knit', 'breezy.bzr.remote', 'breezy.bzr.smart.client', 'breezy.bzr.workingtree_4', 'breezy.bzr.xml_serializer', 'breezy.bzr.xml8', 'getpass', 'kerberos', 'merge3', 'smtplib', 'tarfile', 'tempfile', 'termios', 'tty'] + old_format_modules)