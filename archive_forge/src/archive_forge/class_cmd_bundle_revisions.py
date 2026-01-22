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
class cmd_bundle_revisions(cmd_send):
    __doc__ = 'Create a merge-directive for submitting changes.\n\n    A merge directive provides many things needed for requesting merges:\n\n    * A machine-readable description of the merge to perform\n\n    * An optional patch that is a preview of the changes requested\n\n    * An optional bundle of revision data, so that the changes can be applied\n      directly from the merge directive, without retrieving data from a\n      branch.\n\n    If --no-bundle is specified, then public_branch is needed (and must be\n    up-to-date), so that the receiver can perform the merge using the\n    public_branch.  The public_branch is always included if known, so that\n    people can check it later.\n\n    The submit branch defaults to the parent, but can be overridden.  Both\n    submit branch and public branch will be remembered if supplied.\n\n    If a public_branch is known for the submit_branch, that public submit\n    branch is used in the merge instructions.  This means that a local mirror\n    can be used as your actual submit branch, once you have set public_branch\n    for that mirror.\n    '
    takes_options = [Option('no-bundle', help='Do not include a bundle in the merge directive.'), Option('no-patch', help='Do not include a preview patch in the merge directive.'), Option('remember', help='Remember submit and public branch.'), Option('from', help='Branch to generate the submission from, rather than the one containing the working directory.', short_name='f', type=str), Option('output', short_name='o', help='Write directive to this file.', type=str), Option('strict', help='Refuse to bundle revisions if there are uncommitted changes in the working tree, --no-strict disables the check.'), 'revision', RegistryOption('format', help='Use the specified output format.', lazy_registry=('breezy.send', 'format_registry'))]
    aliases = ['bundle']
    _see_also = ['send', 'merge']
    hidden = True

    def run(self, submit_branch=None, public_branch=None, no_bundle=False, no_patch=False, revision=None, remember=False, output=None, format=None, strict=None, **kwargs):
        if output is None:
            output = '-'
        from .send import send
        return send(submit_branch, revision, public_branch, remember, format, no_bundle, no_patch, output, kwargs.get('from', '.'), None, None, None, self.outf, strict=strict)