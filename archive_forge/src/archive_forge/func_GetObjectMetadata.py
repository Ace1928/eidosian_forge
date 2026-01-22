from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import six
from gslib.cloud_api import ServiceException
from gslib.discard_messages_queue import DiscardMessagesQueue
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.translation_helper import CreateBucketNotFoundException
from gslib.utils.translation_helper import CreateObjectNotFoundException
def GetObjectMetadata(self, bucket_name, object_name, generation=None, provider=None, fields=None):
    """See CloudApi class for function doc strings."""
    if generation:
        generation = long(generation)
    if bucket_name in self.buckets:
        bucket = self.buckets[bucket_name]
        if object_name in bucket.objects and bucket.objects[object_name]:
            if generation:
                if 'versioned' in bucket.objects[object_name]:
                    for obj in bucket.objects[object_name]['versioned']:
                        if obj.root_object.generation == generation:
                            return obj.root_object
                if 'live' in bucket.objects[object_name]:
                    if bucket.objects[object_name]['live'].root_object.generation == generation:
                        return bucket.objects[object_name]['live'].root_object
            elif 'live' in bucket.objects[object_name]:
                return bucket.objects[object_name]['live'].root_object
        raise CreateObjectNotFoundException(404, self.provider, bucket_name, object_name)
    raise CreateBucketNotFoundException(404, self.provider, bucket_name)