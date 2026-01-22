import errno
import os
import sys
import breezy.bzr
import breezy.git
from . import controldir, errors, lazy_import, transport
import time
import breezy
from breezy import (
from breezy.branch import Branch
from breezy.transport import memory
from breezy.smtp_connection import SMTPConnection
from breezy.workingtree import WorkingTree
from breezy.i18n import gettext, ngettext
from .commands import Command, builtin_command_registry, display_command
from .option import (ListOption, Option, RegistryOption, _parse_revision_str,
from .revisionspec import RevisionInfo, RevisionSpec
from .trace import get_verbosity_level, is_quiet, mutter, note, warning
class cmd_ignore(Command):
    __doc__ = 'Ignore specified files or patterns.\n\n    See ``brz help patterns`` for details on the syntax of patterns.\n\n    If a .bzrignore file does not exist, the ignore command\n    will create one and add the specified files or patterns to the newly\n    created file. The ignore command will also automatically add the\n    .bzrignore file to be versioned. Creating a .bzrignore file without\n    the use of the ignore command will require an explicit add command.\n\n    To remove patterns from the ignore list, edit the .bzrignore file.\n    After adding, editing or deleting that file either indirectly by\n    using this command or directly by using an editor, be sure to commit\n    it.\n\n    Breezy also supports a global ignore file ~/.config/breezy/ignore. On\n    Windows the global ignore file can be found in the application data\n    directory as\n    C:\\Documents and Settings\\<user>\\Application Data\\Breezy\\3.0\\ignore.\n    Global ignores are not touched by this command. The global ignore file\n    can be edited directly using an editor.\n\n    Patterns prefixed with \'!\' are exceptions to ignore patterns and take\n    precedence over regular ignores.  Such exceptions are used to specify\n    files that should be versioned which would otherwise be ignored.\n\n    Patterns prefixed with \'!!\' act as regular ignore patterns, but have\n    precedence over the \'!\' exception patterns.\n\n    :Notes:\n\n    * Ignore patterns containing shell wildcards must be quoted from\n      the shell on Unix.\n\n    * Ignore patterns starting with "#" act as comments in the ignore file.\n      To ignore patterns that begin with that character, use the "RE:" prefix.\n\n    :Examples:\n        Ignore the top level Makefile::\n\n            brz ignore ./Makefile\n\n        Ignore .class files in all directories...::\n\n            brz ignore "*.class"\n\n        ...but do not ignore "special.class"::\n\n            brz ignore "!special.class"\n\n        Ignore files whose name begins with the "#" character::\n\n            brz ignore "RE:^#"\n\n        Ignore .o files under the lib directory::\n\n            brz ignore "lib/**/*.o"\n\n        Ignore .o files under the lib directory::\n\n            brz ignore "RE:lib/.*\\.o"\n\n        Ignore everything but the "debian" toplevel directory::\n\n            brz ignore "RE:(?!debian/).*"\n\n        Ignore everything except the "local" toplevel directory,\n        but always ignore autosave files ending in ~, even under local/::\n\n            brz ignore "*"\n            brz ignore "!./local"\n            brz ignore "!!*~"\n    '
    _see_also = ['status', 'ignored', 'patterns']
    takes_args = ['name_pattern*']
    takes_options = ['directory', Option('default-rules', help='Display the default ignore rules that brz uses.')]

    def run(self, name_pattern_list=None, default_rules=None, directory='.'):
        from breezy import ignores
        if default_rules is not None:
            for pattern in ignores.USER_DEFAULTS:
                self.outf.write('%s\n' % pattern)
            return
        if not name_pattern_list:
            raise errors.CommandError(gettext('ignore requires at least one NAME_PATTERN or --default-rules.'))
        name_pattern_list = [globbing.normalize_pattern(p) for p in name_pattern_list]
        bad_patterns = ''
        bad_patterns_count = 0
        for p in name_pattern_list:
            if not globbing.Globster.is_pattern_valid(p):
                bad_patterns_count += 1
                bad_patterns += '\n  %s' % p
        if bad_patterns:
            msg = ngettext('Invalid ignore pattern found. %s', 'Invalid ignore patterns found. %s', bad_patterns_count) % bad_patterns
            ui.ui_factory.show_error(msg)
            raise lazy_regex.InvalidPattern('')
        for name_pattern in name_pattern_list:
            if name_pattern[0] == '/' or (len(name_pattern) > 1 and name_pattern[1] == ':'):
                raise errors.CommandError(gettext('NAME_PATTERN should not be an absolute path'))
        tree, relpath = WorkingTree.open_containing(directory)
        ignores.tree_ignores_add_patterns(tree, name_pattern_list)
        ignored = globbing.Globster(name_pattern_list)
        matches = []
        self.enter_context(tree.lock_read())
        for filename, fc, fkind, entry in tree.list_files():
            id = getattr(entry, 'file_id', None)
            if id is not None:
                if ignored.match(filename):
                    matches.append(filename)
        if len(matches) > 0:
            self.outf.write(gettext("Warning: the following files are version controlled and match your ignore pattern:\n%s\nThese files will continue to be version controlled unless you 'brz remove' them.\n") % ('\n'.join(matches),))