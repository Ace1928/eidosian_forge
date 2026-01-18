from . import controldir, errors, gpg
from . import repository as _mod_repository
from . import revision as _mod_revision
from .commands import Command
from .i18n import gettext, ngettext
from .option import Option
def write_verbose(string):
    self.outf.write('  ' + string + '\n')