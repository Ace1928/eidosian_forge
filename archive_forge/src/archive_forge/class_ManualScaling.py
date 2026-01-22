from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManualScaling(_messages.Message):
    """A service with manual scaling runs continuously, allowing you to perform
  complex initialization and rely on the state of its memory over time.

  Fields:
    instances: Number of instances to assign to the service at the start. This
      number can later be altered by using the Modules API
      (https://cloud.google.com/appengine/docs/python/modules/functions)
      set_num_instances() function.
  """
    instances = _messages.IntegerField(1, variant=_messages.Variant.INT32)