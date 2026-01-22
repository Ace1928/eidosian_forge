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
class BzrGitMappingv1(BzrGitMapping):
    revid_prefix = b'git-v1'
    experimental = False

    def __str__(self):
        return self.revid_prefix.decode('utf-8')