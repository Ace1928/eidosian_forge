from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Http(_messages.Message):
    """Defines the HTTP configuration for a service. It contains a list of
  HttpRule, each specifying the mapping of an RPC method to one or more HTTP
  REST API methods.

  Fields:
    rules: A list of HTTP rules for configuring the HTTP REST API methods.
  """
    rules = _messages.MessageField('HttpRule', 1, repeated=True)