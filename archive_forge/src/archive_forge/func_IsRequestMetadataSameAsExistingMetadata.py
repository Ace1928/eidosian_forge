from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
def IsRequestMetadataSameAsExistingMetadata(request_metadata, existing_metadata):
    for key, value in request_metadata.items():
        if key not in existing_metadata or value != existing_metadata[key]:
            return False
    return True