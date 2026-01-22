from io import BytesIO
from typing import TYPE_CHECKING, Optional, Union
from ..lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
from .. import errors, lockable_files
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from .. import urlutils
from ..branch import (Branch, BranchFormat, BranchWriteLockResult,
from ..controldir import ControlDir
from ..decorators import only_raises
from ..lock import LogicalLockResult, _RelockDebugMixin
from ..trace import mutter
from . import bzrdir, rio
from .repository import MetaDirRepository
class BzrBranch8(BzrBranch):
    """A branch that stores tree-reference locations."""

    def _open_hook(self, possible_transports=None):
        if self._ignore_fallbacks:
            return
        if possible_transports is None:
            possible_transports = [self.controldir.root_transport]
        try:
            url = self.get_stacked_on_url()
        except (errors.UnstackableRepositoryFormat, errors.NotStacked, UnstackableBranchFormat):
            pass
        else:
            for hook in Branch.hooks['transform_fallback_location']:
                url = hook(self, url)
                if url is None:
                    hook_name = Branch.hooks.get_hook_name(hook)
                    raise AssertionError("'transform_fallback_location' hook %s returned None, not a URL." % hook_name)
            self._activate_fallback_location(url, possible_transports=possible_transports)

    def __init__(self, *args, **kwargs):
        self._ignore_fallbacks = kwargs.get('ignore_fallbacks', False)
        super().__init__(*args, **kwargs)
        self._last_revision_info_cache = None
        self._reference_info = None

    def _clear_cached_state(self):
        super()._clear_cached_state()
        self._last_revision_info_cache = None
        self._reference_info = None

    def _check_history_violation(self, revision_id):
        last_revision = self.last_revision()
        if _mod_revision.is_null(last_revision):
            return
        graph = self.repository.get_graph()
        for lh_ancestor in graph.iter_lefthand_ancestry(revision_id):
            if lh_ancestor == last_revision:
                return
        raise errors.AppendRevisionsOnlyViolation(self.user_url)

    def _gen_revision_history(self):
        """Generate the revision history from last revision
        """
        last_revno, last_revision = self.last_revision_info()
        self._extend_partial_history(stop_index=last_revno - 1)
        return list(reversed(self._partial_revision_history_cache))

    def _set_parent_location(self, url):
        """Set the parent branch"""
        with self.lock_write():
            self._set_config_location('parent_location', url, make_relative=True)

    def _get_parent_location(self):
        """Set the parent branch"""
        with self.lock_read():
            return self._get_config_location('parent_location')

    def _set_all_reference_info(self, info_dict):
        """Replace all reference info stored in a branch.

        :param info_dict: A dict of {file_id: (branch_location, tree_path)}
        """
        s = BytesIO()
        writer = rio.RioWriter(s)
        for file_id, (branch_location, tree_path) in info_dict.items():
            stanza = rio.Stanza(file_id=file_id, branch_location=branch_location)
            if tree_path is not None:
                stanza.add('tree_path', tree_path)
            writer.write_stanza(stanza)
        with self.lock_write():
            self._transport.put_bytes('references', s.getvalue())
            self._reference_info = info_dict

    def _get_all_reference_info(self):
        """Return all the reference info stored in a branch.

        :return: A dict of {tree_path: (branch_location, file_id)}
        """
        with self.lock_read():
            if self._reference_info is not None:
                return self._reference_info
            try:
                with self._transport.get('references') as rio_file:
                    stanzas = rio.read_stanzas(rio_file)
                    info_dict = {s['file_id'].encode('utf-8'): (s['branch_location'], s['tree_path'] if 'tree_path' in s else None) for s in stanzas}
            except _mod_transport.NoSuchFile:
                info_dict = {}
            self._reference_info = info_dict
            return info_dict

    def set_reference_info(self, file_id, branch_location, tree_path=None):
        """Set the branch location to use for a tree reference.

        :param branch_location: The location of the branch to retrieve tree
            references from.
        :param file_id: The file-id of the tree reference.
        :param tree_path: The path of the tree reference in the tree.
        """
        info_dict = self._get_all_reference_info()
        info_dict[file_id] = (branch_location, tree_path)
        if branch_location is None:
            del info_dict[file_id]
        self._set_all_reference_info(info_dict)

    def get_reference_info(self, file_id):
        """Get the tree_path and branch_location for a tree reference.

        :return: a tuple of (branch_location, tree_path)
        """
        return self._get_all_reference_info().get(file_id, (None, None))

    def set_push_location(self, location):
        """See Branch.set_push_location."""
        self._set_config_location('push_location', location)

    def set_bound_location(self, location):
        """See Branch.set_push_location."""
        self._master_branch_cache = None
        conf = self.get_config_stack()
        if location is None:
            if not conf.get('bound'):
                return False
            else:
                conf.set('bound', 'False')
                return True
        else:
            self._set_config_location('bound_location', location, config=conf)
            conf.set('bound', 'True')
        return True

    def _get_bound_location(self, bound):
        """Return the bound location in the config file.

        Return None if the bound parameter does not match"""
        conf = self.get_config_stack()
        if conf.get('bound') != bound:
            return None
        return self._get_config_location('bound_location', config=conf)

    def get_bound_location(self):
        """See Branch.get_bound_location."""
        return self._get_bound_location(True)

    def get_old_bound_location(self):
        """See Branch.get_old_bound_location"""
        return self._get_bound_location(False)

    def get_stacked_on_url(self):
        conf = _mod_config.BranchOnlyStack(self)
        stacked_url = self._get_config_location('stacked_on_location', config=conf)
        if stacked_url is None:
            raise errors.NotStacked(self)
        return stacked_url

    def get_rev_id(self, revno, history=None):
        """Find the revision id of the specified revno."""
        if revno == 0:
            return _mod_revision.NULL_REVISION
        with self.lock_read():
            last_revno, last_revision_id = self.last_revision_info()
            if revno <= 0 or revno > last_revno:
                raise errors.RevnoOutOfBounds(revno, (0, last_revno))
            if history is not None:
                return history[revno - 1]
            index = last_revno - revno
            if len(self._partial_revision_history_cache) <= index:
                self._extend_partial_history(stop_index=index)
            if len(self._partial_revision_history_cache) > index:
                return self._partial_revision_history_cache[index]
            else:
                raise errors.NoSuchRevision(self, revno)

    def revision_id_to_revno(self, revision_id):
        """Given a revision id, return its revno"""
        if _mod_revision.is_null(revision_id):
            return 0
        with self.lock_read():
            try:
                index = self._partial_revision_history_cache.index(revision_id)
            except ValueError:
                try:
                    self._extend_partial_history(stop_revision=revision_id)
                except errors.RevisionNotPresent as exc:
                    raise errors.GhostRevisionsHaveNoRevno(revision_id, exc.revision_id) from exc
                index = len(self._partial_revision_history_cache) - 1
                if index < 0:
                    raise errors.NoSuchRevision(self, revision_id)
                if self._partial_revision_history_cache[index] != revision_id:
                    raise errors.NoSuchRevision(self, revision_id)
            return self.revno() - index