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
class SmartServerRepositorySetMakeWorkingTrees(SmartServerRepositoryRequest):

    def do_repository_request(self, repository, str_bool_new_value):
        if str_bool_new_value == b'True':
            new_value = True
        else:
            new_value = False
        repository.set_make_working_trees(new_value)
        return SuccessfulSmartServerResponse((b'ok',))