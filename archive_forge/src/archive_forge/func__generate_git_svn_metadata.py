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
def _generate_git_svn_metadata(self, rev, encoding):
    try:
        git_svn_id = rev.properties['git-svn-id']
    except KeyError:
        return ''
    else:
        return '\ngit-svn-id: %s\n' % git_svn_id.encode(encoding)