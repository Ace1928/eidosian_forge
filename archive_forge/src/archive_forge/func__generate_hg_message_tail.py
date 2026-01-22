import base64
import stat
from typing import Optional
import fastbencode as bencode
from .. import errors, foreign, trace, urlutils
from ..foreign import ForeignRevision, ForeignVcs, VcsMappingRegistry
from ..revision import NULL_REVISION, Revision
from .errors import NoPushSupport
from .hg import extract_hg_metadata, format_hg_metadata
from .roundtrip import (CommitSupplement, extract_bzr_metadata,
def _generate_hg_message_tail(self, rev):
    extra = {}
    renames = []
    branch = 'default'
    for name in rev.properties:
        if name == 'hg:extra:branch':
            branch = rev.properties['hg:extra:branch']
        elif name.startswith('hg:extra'):
            extra[name[len('hg:extra:'):]] = base64.b64decode(rev.properties[name])
        elif name == 'hg:renames':
            renames = bencode.bdecode(base64.b64decode(rev.properties['hg:renames']))
    ret = format_hg_metadata(renames, branch, extra)
    if not isinstance(ret, bytes):
        raise TypeError(ret)
    return ret