from the atom and gd namespaces. For more information, see:
from __future__ import absolute_import
import base64
import calendar
import datetime
import os
import re
import time
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import namespace_manager
from googlecloudsdk.third_party.appengine.api import users
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import entity_v4_pb
from googlecloudsdk.third_party.appengine.datastore import sortable_pb_encoder
from googlecloudsdk.third_party.appengine._internal import six_subset
class EmbeddedEntity(_BaseByteType):
    """A proto encoded EntityProto.

  This behaves identically to Blob, except for the
  constructor, which accepts a str or EntityProto argument.

  Can be decoded using datastore.Entity.FromProto(), db.model_from_protobuf() or
  ndb.LocalStructuredProperty.
  """

    def __new__(cls, arg=None):
        """Constructor.

    Args:
      arg: optional str or EntityProto instance (default '')
    """
        if isinstance(arg, entity_pb.EntityProto):
            arg = arg.SerializePartialToString()
        return super(EmbeddedEntity, cls).__new__(cls, arg)