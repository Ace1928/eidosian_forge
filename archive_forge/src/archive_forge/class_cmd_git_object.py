import breezy.bzr  # noqa: F401
from breezy import controldir
from ..commands import Command, display_command
from ..option import Option, RegistryOption
class cmd_git_object(Command):
    """List or display Git objects by SHA.

    Cat a particular object's Git representation if a SHA is specified.
    List all available SHAs otherwise.
    """
    hidden = True
    aliases = ['git-objects', 'git-cat']
    takes_args = ['sha1?']
    takes_options = [Option('directory', short_name='d', help='Location of repository.', type=str), Option('pretty', help='Pretty-print objects.')]
    encoding_type = 'exact'

    @display_command
    def run(self, sha1=None, directory='.', pretty=False):
        from ..controldir import ControlDir
        from ..errors import CommandError
        from ..i18n import gettext
        from .object_store import get_object_store
        controldir, _ = ControlDir.open_containing(directory)
        repo = controldir.find_repository()
        object_store = get_object_store(repo)
        with object_store.lock_read():
            if sha1 is not None:
                try:
                    obj = object_store[sha1.encode('ascii')]
                except KeyError:
                    raise CommandError(gettext('Object not found: %s') % sha1)
                if pretty:
                    text = obj.as_pretty_string()
                else:
                    text = obj.as_raw_string()
                self.outf.write(text)
            else:
                for sha1 in object_store:
                    self.outf.write('%s\n' % sha1.decode('ascii'))