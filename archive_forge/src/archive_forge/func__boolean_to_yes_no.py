import fastbencode as bencode
from ... import branch, errors, repository, urlutils
from ...controldir import network_format_registry
from .. import BzrProber
from ..bzrdir import BzrDir, BzrDirFormat
from .request import (FailedSmartServerResponse, SmartServerRequest,
def _boolean_to_yes_no(self, a_boolean):
    if a_boolean:
        return b'yes'
    else:
        return b'no'