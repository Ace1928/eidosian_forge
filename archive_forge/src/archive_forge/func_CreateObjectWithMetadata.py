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
def CreateObjectWithMetadata(self, apitools_object, contents=''):
    """Creates an object in the bucket according to the input metadata.

    This will create a new object version (ignoring the generation specified
    in the input object).

    Args:
      apitools_object: apitools Object.
      contents: optional object contents.

    Returns:
      apitools Object representing created object.
    """
    object_name = apitools_object.name
    if self.root_object.versioning and self.root_object.versioning.enabled and (apitools_object.name in self.objects):
        if 'live' in self.objects[object_name]:
            apitools_object.generation = self.objects[object_name]['live'].root_object.generation + 1
            if 'versioned' not in self.objects[object_name]:
                self.objects[object_name]['versioned'] = []
            self.objects[object_name]['versioned'].append(self.objects[object_name]['live'])
        elif 'versioned' in self.objects[object_name] and self.objects[object_name]['versioned']:
            apitools_object.generation = self.objects[object_name]['versioned'][-1].root_object.generation + 1
    else:
        apitools_object.generation = 1
        self.objects[object_name] = {}
    new_object = MockObject(apitools_object, contents=contents)
    self.objects[object_name]['live'] = new_object
    return new_object