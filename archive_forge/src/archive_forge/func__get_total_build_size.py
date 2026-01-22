import operator
import os
from io import BytesIO
from ..lazy_import import lazy_import
import patiencediff
import gzip
from breezy import (
from breezy.bzr import (
from breezy.bzr import pack_repo
from breezy.i18n import gettext
from .. import annotate, errors, osutils
from .. import transport as _mod_transport
from ..bzr.versionedfile import (AbsentContentFactory, ConstantMapper,
from ..errors import InternalBzrError, InvalidRevisionId, RevisionNotPresent
from ..osutils import contains_whitespace, sha_string, sha_strings, split_lines
from ..transport import NoSuchFile
from . import index as _mod_index
def _get_total_build_size(self, keys, positions):
    """Determine the total bytes to build these keys.

    (helper function because _KnitGraphIndex and _KndxIndex work the same, but
    don't inherit from a common base.)

    :param keys: Keys that we want to build
    :param positions: dict of {key, (info, index_memo, comp_parent)} (such
        as returned by _get_components_positions)
    :return: Number of bytes to build those keys
    """
    all_build_index_memos = {}
    build_keys = keys
    while build_keys:
        next_keys = set()
        for key in build_keys:
            if key not in positions:
                continue
            _, index_memo, compression_parent = positions[key]
            all_build_index_memos[key] = index_memo
            if compression_parent not in all_build_index_memos:
                next_keys.add(compression_parent)
        build_keys = next_keys
    return sum((index_memo[2] for index_memo in all_build_index_memos.values()))