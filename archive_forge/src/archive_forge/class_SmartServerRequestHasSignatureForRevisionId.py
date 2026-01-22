import bz2
import itertools
import os
import queue
import sys
import tempfile
import threading
import zlib
import fastbencode as bencode
from ... import errors, estimate_compressed_size, osutils
from ... import revision as _mod_revision
from ... import trace, ui
from ...repository import _strip_NULL_ghosts, network_format_registry
from .. import inventory as _mod_inventory
from .. import inventory_delta, pack, vf_search
from ..bzrdir import BzrDir
from ..versionedfile import (ChunkedContentFactory, NetworkRecordStream,
from .request import (FailedSmartServerResponse, SmartServerRequest,
class SmartServerRequestHasSignatureForRevisionId(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, revision_id):
        """Return ok if a signature is present for a revision.

        Introduced in bzr 2.5.0.

        :param repository: The repository to query in.
        :param revision_id: The utf8 encoded revision_id to lookup.
        :return: A smart server response of ('yes', ) if a
            signature for the revision is present,
            ('no', ) if it is missing.
        """
        try:
            if repository.has_signature_for_revision_id(revision_id):
                return SuccessfulSmartServerResponse((b'yes',))
            else:
                return SuccessfulSmartServerResponse((b'no',))
        except errors.NoSuchRevision:
            return FailedSmartServerResponse((b'nosuchrevision', revision_id))