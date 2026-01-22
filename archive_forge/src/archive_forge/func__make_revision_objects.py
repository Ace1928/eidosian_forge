import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def _make_revision_objects(branch, generate_delta, search, log_rev_iterator):
    """Extract revision objects from the repository

    :param branch: The branch being logged.
    :param generate_delta: Whether to generate a delta for each revision.
    :param search: A user text search string.
    :param log_rev_iterator: An input iterator containing all revisions that
        could be displayed, in lists.
    :return: An iterator over lists of ((rev_id, revno, merge_depth), rev,
        delta).
    """
    repository = branch.repository
    for revs in log_rev_iterator:
        revision_ids = [view[0] for view, _, _ in revs]
        revisions = dict(repository.iter_revisions(revision_ids))
        yield [(rev[0], revisions[rev[0][0]], rev[2]) for rev in revs]