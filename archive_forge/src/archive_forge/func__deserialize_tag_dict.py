import fastbencode as bencode
from .. import errors, trace
from .. import transport as _mod_transport
from ..tag import Tags
def _deserialize_tag_dict(self, tag_content):
    """Convert the tag file into a dictionary of tags"""
    if tag_content == b'':
        return {}
    try:
        r = {}
        for k, v in bencode.bdecode(tag_content).items():
            r[k.decode('utf-8')] = v
        return r
    except ValueError as e:
        raise ValueError('failed to deserialize tag dictionary %r: %s' % (tag_content, e))