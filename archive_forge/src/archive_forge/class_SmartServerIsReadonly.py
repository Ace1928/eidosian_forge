import threading
from _thread import get_ident
from ... import branch as _mod_branch
from ... import debug, errors, osutils, registry, revision, trace
from ... import transport as _mod_transport
from ... import urlutils
from ...lazy_import import lazy_import
from breezy.bzr import bzrdir
from breezy.bzr.bundle import serializer
import tempfile
class SmartServerIsReadonly(SmartServerRequest):

    def do(self):
        if self._backing_transport.is_readonly():
            answer = b'yes'
        else:
            answer = b'no'
        return SuccessfulSmartServerResponse((answer,))