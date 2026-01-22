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
class BzrDirMetaFormat1(BzrDirFormat):
    """Bzr meta control format 1

    This is the first format with split out working tree, branch and repository
    disk storage.

    It has:

    - Format 3 working trees [optional]
    - Format 5 branches [optional]
    - Format 7 repositories [optional]
    """
    _lock_class = lockdir.LockDir
    fixed_components = False
    colocated_branches = True

    def __init__(self):
        BzrDirFormat.__init__(self)
        self._workingtree_format = None
        self._branch_format = None
        self._repository_format = None

    def __eq__(self, other):
        if other.__class__ is not self.__class__:
            return False
        if other.repository_format != self.repository_format:
            return False
        if other.workingtree_format != self.workingtree_format:
            return False
        if other.features != self.features:
            return False
        return True

    def __ne__(self, other):
        return not self == other

    def get_branch_format(self):
        if self._branch_format is None:
            from .branch import format_registry as branch_format_registry
            self._branch_format = branch_format_registry.get_default()
        return self._branch_format

    def set_branch_format(self, format):
        self._branch_format = format

    def require_stacking(self, stack_on=None, possible_transports=None, _skip_repo=False):
        """We have a request to stack, try to ensure the formats support it.

        :param stack_on: If supplied, it is the URL to a branch that we want to
            stack on. Check to see if that format supports stacking before
            forcing an upgrade.
        """
        new_repo_format = None
        new_branch_format = None
        target = [None, False, None]

        def get_target_branch():
            if target[1]:
                return target
            if stack_on is None:
                target[:] = [None, True, True]
                return target
            try:
                target_dir = BzrDir.open(stack_on, possible_transports=possible_transports)
            except errors.NotBranchError:
                target[:] = [None, True, False]
                return target
            except errors.JailBreak:
                target[:] = [None, True, True]
                return target
            try:
                target_branch = target_dir.open_branch()
            except errors.NotBranchError:
                target[:] = [None, True, False]
                return target
            target[:] = [target_branch, True, False]
            return target
        if not _skip_repo and (not self.repository_format.supports_external_lookups):
            target_branch, _, do_upgrade = get_target_branch()
            if target_branch is None:
                if do_upgrade:
                    if self.repository_format.rich_root_data:
                        new_repo_format = knitpack_repo.RepositoryFormatKnitPack5RichRoot()
                    else:
                        new_repo_format = knitpack_repo.RepositoryFormatKnitPack5()
            else:
                new_repo_format = target_branch.repository._format
                if not new_repo_format.supports_external_lookups:
                    new_repo_format = None
            if new_repo_format is not None:
                self.repository_format = new_repo_format
                note(gettext('Source repository format does not support stacking, using format:\n  %s'), new_repo_format.get_format_description())
        if not self.get_branch_format().supports_stacking():
            target_branch, _, do_upgrade = get_target_branch()
            if target_branch is None:
                if do_upgrade:
                    from .branch import BzrBranchFormat7
                    new_branch_format = BzrBranchFormat7()
            else:
                new_branch_format = target_branch._format
                if not new_branch_format.supports_stacking():
                    new_branch_format = None
            if new_branch_format is not None:
                self.set_branch_format(new_branch_format)
                note(gettext('Source branch format does not support stacking, using format:\n  %s'), new_branch_format.get_format_description())

    def get_converter(self, format=None):
        """See BzrDirFormat.get_converter()."""
        if format is None:
            format = BzrDirFormat.get_default_format()
        if isinstance(self, BzrDirMetaFormat1) and isinstance(format, BzrDirMetaFormat1Colo):
            return ConvertMetaToColo(format)
        if isinstance(self, BzrDirMetaFormat1Colo) and isinstance(format, BzrDirMetaFormat1):
            return ConvertMetaToColo(format)
        if not isinstance(self, format.__class__):
            raise NotImplementedError(self.get_converter)
        return ConvertMetaToMeta(format)

    @classmethod
    def get_format_string(cls):
        """See BzrDirFormat.get_format_string()."""
        return b'Bazaar-NG meta directory, format 1\n'

    def get_format_description(self):
        """See BzrDirFormat.get_format_description()."""
        return 'Meta directory format 1'

    def _open(self, transport):
        """See BzrDirFormat._open."""
        format = BzrDirMetaFormat1()
        self._supply_sub_formats_to(format)
        return BzrDirMeta1(transport, format)

    def __return_repository_format(self):
        """Circular import protection."""
        if self._repository_format:
            return self._repository_format
        from .repository import format_registry
        return format_registry.get_default()

    def _set_repository_format(self, value):
        """Allow changing the repository format for metadir formats."""
        self._repository_format = value
    repository_format = property(__return_repository_format, _set_repository_format)

    def _supply_sub_formats_to(self, other_format):
        """Give other_format the same values for sub formats as this has.

        This method is expected to be used when parameterising a
        RemoteBzrDirFormat instance with the parameters from a
        BzrDirMetaFormat1 instance.

        :param other_format: other_format is a format which should be
            compatible with whatever sub formats are supported by self.
        :return: None.
        """
        super()._supply_sub_formats_to(other_format)
        if getattr(self, '_repository_format', None) is not None:
            other_format.repository_format = self.repository_format
        if self._branch_format is not None:
            other_format._branch_format = self._branch_format
        if self._workingtree_format is not None:
            other_format.workingtree_format = self.workingtree_format

    def __get_workingtree_format(self):
        if self._workingtree_format is None:
            from .workingtree import format_registry as wt_format_registry
            self._workingtree_format = wt_format_registry.get_default()
        return self._workingtree_format

    def __set_workingtree_format(self, wt_format):
        self._workingtree_format = wt_format

    def __repr__(self):
        return '<{!r}>'.format(self.__class__.__name__)
    workingtree_format = property(__get_workingtree_format, __set_workingtree_format)