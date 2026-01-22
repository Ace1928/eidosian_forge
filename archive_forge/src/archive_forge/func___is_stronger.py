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
def __is_stronger(self, **kwargs):
    """Internal helper to ask whether a configuration is stronger than another.

    A configuration is stronger when it contains every name/value pair in
    kwargs.

    Example: a configuration with:
      (deadline=5, on_configuration=None, read_policy=EVENTUAL_CONSISTENCY)
    is stronger than:
      (deadline=5, on_configuration=None)
    but not stronger than:
      (deadline=5, on_configuration=None, read_policy=None)
    or
      (deadline=10, on_configuration=None, read_policy=None).

    More formally:
      - Any value is stronger than an unset value;
      - Any value is stronger than itself.

    Returns:
      True if each of the self attributes is stronger than the
    corresponding argument.
    """
    for key, value in kwargs.items():
        if key not in self._values or value != self._values[key]:
            return False
    return True