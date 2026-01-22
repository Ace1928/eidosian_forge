import base64
import os
import pprint
from io import BytesIO
from ... import cache_utf8, osutils, timestamp
from ...errors import BzrError, NoSuchId, TestamentMismatch
from ...osutils import pathjoin, sha_string, sha_strings
from ...revision import NULL_REVISION, Revision
from ...trace import mutter, warning
from ...tree import InterTree, Tree
from ..inventory import (Inventory, InventoryDirectory, InventoryFile,
from ..inventorytree import InventoryTree
from ..testament import StrictTestament
from ..xml5 import serializer_v5
from . import apply_bundle
class RevisionInfo:
    """Gets filled out for each revision object that is read.
    """

    def __init__(self, revision_id):
        self.revision_id = revision_id
        self.sha1 = None
        self.committer = None
        self.date = None
        self.timestamp = None
        self.timezone = None
        self.inventory_sha1 = None
        self.parent_ids = None
        self.base_id = None
        self.message = None
        self.properties = None
        self.tree_actions = None

    def __str__(self):
        return pprint.pformat(self.__dict__)

    def as_revision(self):
        rev = Revision(revision_id=self.revision_id, committer=self.committer, timestamp=float(self.timestamp), timezone=int(self.timezone), inventory_sha1=self.inventory_sha1, message='\n'.join(self.message))
        if self.parent_ids:
            rev.parent_ids.extend(self.parent_ids)
        if self.properties:
            for property in self.properties:
                key_end = property.find(': ')
                if key_end == -1:
                    if not property.endswith(':'):
                        raise ValueError(property)
                    key = str(property[:-1])
                    value = ''
                else:
                    key = str(property[:key_end])
                    value = property[key_end + 2:]
                rev.properties[key] = value
        return rev

    @staticmethod
    def from_revision(revision):
        revision_info = RevisionInfo(revision.revision_id)
        date = timestamp.format_highres_date(revision.timestamp, revision.timezone)
        revision_info.date = date
        revision_info.timezone = revision.timezone
        revision_info.timestamp = revision.timestamp
        revision_info.message = revision.message.split('\n')
        revision_info.properties = [': '.join(p) for p in revision.properties.items()]
        return revision_info