from typing import List
from ...bzr import inventory
from ...bzr.inventory import ROOT_ID, Inventory
from ...bzr.xml_serializer import (Element, SubElement, XMLSerializer,
from ...errors import BzrError
from ...revision import Revision
def _unpack_revision(self, elt):
    """XML Element -> Revision object"""
    if elt.tag not in ('revision', 'changeset'):
        raise BzrError('unexpected tag in revision file: %r' % elt)
    rev = Revision(committer=elt.get('committer'), timestamp=float(elt.get('timestamp')), revision_id=elt.get('revision_id'), inventory_id=elt.get('inventory_id'), inventory_sha1=elt.get('inventory_sha1'))
    precursor = elt.get('precursor')
    precursor_sha1 = elt.get('precursor_sha1')
    pelts = elt.find('parents')
    if pelts:
        for p in pelts:
            rev.parent_ids.append(p.get('revision_id'))
            rev.parent_sha1s.append(p.get('revision_sha1'))
        if precursor:
            prec_parent = rev.parent_ids[0]
    elif precursor:
        rev.parent_ids.append(precursor)
        rev.parent_sha1s.append(precursor_sha1)
    v = elt.get('timezone')
    rev.timezone = v and int(v)
    rev.message = elt.findtext('message')
    return rev