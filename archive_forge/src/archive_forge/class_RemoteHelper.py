import os
from ..controldir import ControlDir
from ..errors import NoRepositoryPresent, NotBranchError
from ..plugins.fastimport import exporter as fastexporter
from ..repository import InterRepository
from ..transport import get_transport_from_path
from . import LocalGitProber
from .dir import BareLocalGitControlDirFormat, LocalGitControlDirFormat
from .object_store import get_object_store
from .refs import get_refs_container, ref_to_branch_name
from .repository import GitRepository
class RemoteHelper:
    """Git remote helper."""

    def __init__(self, local_dir, shortname, remote_dir):
        self.local_dir = local_dir
        self.shortname = shortname
        self.remote_dir = remote_dir
        self.batchcmd = None
        self.wants = []

    def cmd_capabilities(self, outf, argv):
        outf.write(b'\n'.join([c.encode() for c in CAPABILITIES]) + b'\n\n')

    def cmd_list(self, outf, argv):
        try:
            repo = self.remote_dir.find_repository()
        except NoRepositoryPresent:
            repo = self.remote_dir.create_repository()
        object_store = get_object_store(repo)
        with object_store.lock_read():
            refs = get_refs_container(self.remote_dir, object_store)
            for ref, git_sha1 in refs.as_dict().items():
                ref = ref.replace(b'~', b'_')
                outf.write(b'%s %s\n' % (git_sha1, ref))
            outf.write(b'\n')

    def cmd_option(self, outf, argv):
        outf.write(b'unsupported\n')

    def cmd_fetch(self, outf, argv):
        if self.batchcmd not in (None, 'fetch'):
            raise Exception('fetch command inside other batch command')
        self.wants.append(tuple(argv[1:]))
        self.batchcmd = 'fetch'

    def cmd_push(self, outf, argv):
        if self.batchcmd not in (None, 'push'):
            raise Exception('push command inside other batch command')
        self.wants.append(tuple(argv[1].split(':', 1)))
        self.batchcmd = 'push'

    def cmd_import(self, outf, argv):
        if 'fastimport' in CAPABILITIES:
            raise Exception("install fastimport for 'import' command support")
        ref = argv[1].encode('utf-8')
        dest_branch_name = ref_to_branch_name(ref)
        if dest_branch_name == 'master':
            dest_branch_name = None
        remote_branch = self.remote_dir.open_branch(name=dest_branch_name)
        exporter = fastexporter.BzrFastExporter(remote_branch, outf=outf, ref=ref, checkpoint=None, import_marks_file=None, export_marks_file=None, revision=None, verbose=None, plain_format=True, rewrite_tags=False)
        exporter.run()
    commands = {'capabilities': cmd_capabilities, 'list': cmd_list, 'option': cmd_option, 'fetch': cmd_fetch, 'push': cmd_push, 'import': cmd_import}

    def process(self, inf, outf):
        while True:
            line = inf.readline()
            if not line:
                break
            self.process_line(line, outf)

    def process_line(self, l, outf):
        argv = l.strip().split()
        if argv == []:
            if self.batchcmd == 'fetch':
                fetch(outf, self.wants, self.shortname, self.remote_dir, self.local_dir)
            elif self.batchcmd == 'push':
                push(outf, self.wants, self.shortname, self.remote_dir, self.local_dir)
            elif self.batchcmd is None:
                return
            else:
                raise AssertionError('invalid batch %r' % self.batchcmd)
            self.batchcmd = None
        else:
            try:
                self.commands[argv[0].decode()](self, outf, argv)
            except KeyError:
                raise Exception('Unknown remote command %r' % argv)
        outf.flush()