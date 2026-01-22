from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def ConstructMetadataDict(metadata_msg):
    res = {}
    if metadata_msg:
        for metadata in metadata_msg.items.additionalProperties:
            res[metadata.key] = metadata.value
    return res