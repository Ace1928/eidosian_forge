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
class SmartServerRepositoryGetSerializerFormat(SmartServerRepositoryRequest):

    def do_repository_request(self, repository):
        """Return the serializer format for this repository.

        New in 2.5.0.

        :param repository: The repository to query
        :return: A smart server response (b'ok', FORMAT)
        """
        serializer = repository.get_serializer_format()
        return SuccessfulSmartServerResponse((b'ok', serializer))