from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
def _check_if_descendant_or_diverged(self, revision_a, revision_b, graph, other_branch):
    """Ensure that revision_b is a descendant of revision_a.

        This is a helper function for update_revisions.

        :raises: DivergedBranches if revision_b has diverged from revision_a.
        Returns: True if revision_b is a descendant of revision_a.
        """
    relation = self._revision_relations(revision_a, revision_b, graph)
    if relation == 'b_descends_from_a':
        return True
    elif relation == 'diverged':
        raise errors.DivergedBranches(self, other_branch)
    elif relation == 'a_descends_from_b':
        return False
    else:
        raise AssertionError('invalid relation: {!r}'.format(relation))