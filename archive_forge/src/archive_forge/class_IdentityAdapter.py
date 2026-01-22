from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
class IdentityAdapter(AbstractAdapter):
    """A concrete adapter that implements the identity mapping.

  This is used as the default when a Connection is created without
  specifying an adapter; that's primarily for testing.
  """

    def __init__(self, id_resolver=None):
        super(IdentityAdapter, self).__init__(id_resolver)

    def pb_to_key(self, pb):
        return pb

    def pb_to_entity(self, pb):
        return pb

    def key_to_pb(self, key):
        return key

    def entity_to_pb(self, entity):
        return entity

    def pb_to_index(self, pb):
        return pb