from typing import List
from ...bzr import inventory
from ...bzr.inventory import ROOT_ID, Inventory
from ...bzr.xml_serializer import (Element, SubElement, XMLSerializer,
from ...errors import BzrError
from ...revision import Revision
def _pack_revision(self, rev):
    """Revision object -> xml tree"""
    root = Element('revision', committer=rev.committer, timestamp='%.9f' % rev.timestamp, revision_id=rev.revision_id, inventory_id=rev.inventory_id, inventory_sha1=rev.inventory_sha1)
    if rev.timezone:
        root.set('timezone', str(rev.timezone))
    root.text = '\n'
    msg = SubElement(root, 'message')
    msg.text = escape_invalid_chars(rev.message)[0]
    msg.tail = '\n'
    if rev.parents:
        pelts = SubElement(root, 'parents')
        pelts.tail = pelts.text = '\n'
        for i, parent_id in enumerate(rev.parents):
            p = SubElement(pelts, 'revision_ref')
            p.tail = '\n'
            p.set('revision_id', parent_id)
            if i < len(rev.parent_sha1s):
                p.set('revision_sha1', rev.parent_sha1s[i])
    return root