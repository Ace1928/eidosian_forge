import contextlib
import sys
from typing import TYPE_CHECKING, Set, cast
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from .. import config, controldir, errors, lockdir
from .. import transport as _mod_transport
from ..trace import mutter, note, warning
from ..transport import do_catching_redirections, local
class ConvertMetaToMeta(controldir.Converter):
    """Converts the components of metadirs."""

    def __init__(self, target_format):
        """Create a metadir to metadir converter.

        :param target_format: The final metadir format that is desired.
        """
        self.target_format = target_format

    def convert(self, to_convert, pb):
        """See Converter.convert()."""
        self.controldir = to_convert
        with ui.ui_factory.nested_progress_bar() as self.pb:
            self.count = 0
            self.total = 1
            self.step('checking repository format')
            try:
                repo = self.controldir.open_repository()
            except errors.NoRepositoryPresent:
                pass
            else:
                repo_fmt = self.target_format.repository_format
                if not isinstance(repo._format, repo_fmt.__class__):
                    from ..repository import CopyConverter
                    ui.ui_factory.note(gettext('starting repository conversion'))
                    if not repo_fmt.supports_overriding_transport:
                        raise AssertionError('Repository in metadir does not support overriding transport')
                    converter = CopyConverter(self.target_format.repository_format)
                    converter.convert(repo, pb)
            for branch in self.controldir.list_branches():
                old = branch._format.__class__
                new = self.target_format.get_branch_format().__class__
                while old != new:
                    if old == fullhistorybranch.BzrBranchFormat5 and new in (_mod_bzrbranch.BzrBranchFormat6, _mod_bzrbranch.BzrBranchFormat7, _mod_bzrbranch.BzrBranchFormat8):
                        branch_converter = _mod_bzrbranch.Converter5to6()
                    elif old == _mod_bzrbranch.BzrBranchFormat6 and new in (_mod_bzrbranch.BzrBranchFormat7, _mod_bzrbranch.BzrBranchFormat8):
                        branch_converter = _mod_bzrbranch.Converter6to7()
                    elif old == _mod_bzrbranch.BzrBranchFormat7 and new is _mod_bzrbranch.BzrBranchFormat8:
                        branch_converter = _mod_bzrbranch.Converter7to8()
                    else:
                        raise errors.BadConversionTarget('No converter', new, branch._format)
                    branch_converter.convert(branch)
                    branch = self.controldir.open_branch()
                    old = branch._format.__class__
            try:
                tree = self.controldir.open_workingtree(recommend_upgrade=False)
            except (errors.NoWorkingTree, errors.NotLocalUrl):
                pass
            else:
                if isinstance(tree, workingtree_3.WorkingTree3) and (not isinstance(tree, workingtree_4.DirStateWorkingTree)) and isinstance(self.target_format.workingtree_format, workingtree_4.DirStateWorkingTreeFormat):
                    workingtree_4.Converter3to4().convert(tree)
                if isinstance(tree, workingtree_4.DirStateWorkingTree) and (not isinstance(tree, workingtree_4.WorkingTree5)) and isinstance(self.target_format.workingtree_format, workingtree_4.WorkingTreeFormat5):
                    workingtree_4.Converter4to5().convert(tree)
                if isinstance(tree, workingtree_4.DirStateWorkingTree) and (not isinstance(tree, workingtree_4.WorkingTree6)) and isinstance(self.target_format.workingtree_format, workingtree_4.WorkingTreeFormat6):
                    workingtree_4.Converter4or5to6().convert(tree)
        return to_convert