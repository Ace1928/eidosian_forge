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
def _log_revision_iterator_using_delta_matching(branch, delta_type, match, levels, specific_files, start_rev_id, end_rev_id, direction, exclude_common_ancestry, limit):
    generate_merge_revisions = levels != 1
    delayed_graph_generation = not specific_files and (limit or start_rev_id or end_rev_id)
    view_revisions = _calc_view_revisions(branch, start_rev_id, end_rev_id, direction, generate_merge_revisions=generate_merge_revisions, delayed_graph_generation=delayed_graph_generation, exclude_common_ancestry=exclude_common_ancestry)
    return make_log_rev_iterator(branch, view_revisions, delta_type, match, files=specific_files, direction=direction)