from io import BytesIO
import fastbencode as bencode
from .. import lazy_import
from breezy.bzr import (
from .. import cache_utf8, errors
from .. import revision as _mod_revision
from . import serializer
class CHKBEncodeSerializer(BEncodeRevisionSerializer1, CHKSerializer):
    """A CHKInventory and BEncode based serializer with 'plain' behaviour."""
    format_num = b'10'