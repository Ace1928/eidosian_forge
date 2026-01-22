from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class To(_messages.Message):
    """To clause specifies the host(s), method(s), path(s) and port(s) that the
  rule applies to.

  Fields:
    hosts: List of hosts.
    methods: List of HTTP request methods.
    paths: List of request paths.
    ports: List of host ports.
  """
    hosts = _messages.StringField(1, repeated=True)
    methods = _messages.StringField(2, repeated=True)
    paths = _messages.StringField(3, repeated=True)
    ports = _messages.IntegerField(4, repeated=True, variant=_messages.Variant.INT32)