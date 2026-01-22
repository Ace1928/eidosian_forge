import calendar
import collections.abc
from datetime import datetime
from datetime import timedelta
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def AllFieldsFromDescriptor(self, message_descriptor):
    """Gets all direct fields of Message Descriptor to FieldMask."""
    self.Clear()
    for field in message_descriptor.fields:
        self.paths.append(field.name)