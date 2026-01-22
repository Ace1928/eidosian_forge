from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.storage import errors
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_util
class S3BucketResource(resource_reference.BucketResource):
    """API-specific subclass for handling metadata."""

    def get_json_dump(self):
        return _get_json_dump(self)