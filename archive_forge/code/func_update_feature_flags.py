import itertools
from .. import errors, lockable_files, lockdir
from .. import revision as _mod_revision
from .. import transport as _mod_transport
from ..repository import Repository, RepositoryFormat, format_registry
from . import bzrdir
def update_feature_flags(self, updated_flags):
    """Update the feature flags for this branch.

        :param updated_flags: Dictionary mapping feature names to necessities
            A necessity can be None to indicate the feature should be removed
        """
    with self.lock_write():
        self._format._update_feature_flags(updated_flags)
        self.control_transport.put_bytes('format', self._format.as_string())