from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudApigeeV1Operation(_messages.Message):
    """Represents the pairing of REST resource path and the actions (verbs)
  allowed on the resource path.

  Fields:
    methods: methods refers to the REST verbs as in
      https://www.w3.org/Protocols/rfc2616/rfc2616-sec9.html. When none
      specified, all verb types are allowed.
    resource: Required. REST resource path associated with the API proxy or
      remote service.
  """
    methods = _messages.StringField(1, repeated=True)
    resource = _messages.StringField(2)