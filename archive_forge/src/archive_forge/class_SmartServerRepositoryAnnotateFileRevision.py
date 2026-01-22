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
class SmartServerRepositoryAnnotateFileRevision(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, revision_id, tree_path, file_id=None, default_revision=None):
        """Stream an archive file for a specific revision.

        :param repository: The repository to stream from.
        :param revision_id: Revision for which to export the tree
        :param tree_path: The path inside the tree
        :param file_id: Optional file_id for the file
        :param default_revision: Default revision
        """
        tree = repository.revision_tree(revision_id)
        with tree.lock_read():
            body = bencode.bencode(list(tree.annotate_iter(tree_path.decode('utf-8'), default_revision)))
            return SuccessfulSmartServerResponse((b'ok',), body=body)