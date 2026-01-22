from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import defaultdict
from collections import namedtuple
import six
from apitools.base.protorpclite import protojson
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
def SerializeBindingsTuple(bindings_tuple):
    """Serializes the BindingsValueListEntry instances in a BindingsTuple.

  This is necessary when passing instances of BindingsTuple through
  Command.Apply, as apitools_messages classes are not by default pickleable.

  Args:
    bindings_tuple: A BindingsTuple instance to be serialized.

  Returns:
    A serialized BindingsTuple object.
  """
    return (bindings_tuple.is_grant, [protojson.encode_message(t) for t in bindings_tuple.bindings])