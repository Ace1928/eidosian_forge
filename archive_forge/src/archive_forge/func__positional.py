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
def _positional(max_pos_args):
    """A decorator to declare that only the first N arguments may be positional.

  Note that for methods, n includes 'self'.
  """

    def positional_decorator(wrapped):

        @functools.wraps(wrapped)
        def positional_wrapper(*args, **kwds):
            if len(args) > max_pos_args:
                plural_s = ''
                if max_pos_args != 1:
                    plural_s = 's'
                raise TypeError('%s() takes at most %d positional argument%s (%d given)' % (wrapped.__name__, max_pos_args, plural_s, len(args)))
            return wrapped(*args, **kwds)
        return positional_wrapper
    return positional_decorator