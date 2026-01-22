import collections
import contextlib
import json
import six
from apitools.base.protorpclite import descriptor
from apitools.base.protorpclite import messages
from apitools.gen import extended_descriptor
from apitools.gen import util
@staticmethod
def __ComputeLabel(attrs):
    if attrs.get('required', False):
        return descriptor.FieldDescriptor.Label.REQUIRED
    elif attrs.get('type') == 'array':
        return descriptor.FieldDescriptor.Label.REPEATED
    elif attrs.get('repeated'):
        return descriptor.FieldDescriptor.Label.REPEATED
    return descriptor.FieldDescriptor.Label.OPTIONAL