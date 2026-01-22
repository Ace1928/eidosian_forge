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
class MockObject(object):
    """Defines a mock cloud storage provider object."""

    def __init__(self, root_object, contents=''):
        self.root_object = root_object
        self.contents = contents

    def __str__(self):
        return '%s/%s#%s' % (self.root_object.bucket, self.root_object.name, self.root_object.generation)

    def __repr__(self):
        return str(self)