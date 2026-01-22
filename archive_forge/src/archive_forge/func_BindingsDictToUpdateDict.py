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
def BindingsDictToUpdateDict(bindings):
    """Reformats policy bindings metadata.

  Args:
    bindings: List of dictionaries representing BindingsValueListEntry
      instances. e.g.:
      {
        "role": "some_role",
        "members": ["allAuthenticatedUsers", ...]
      }

  Returns:
    A {role: set(members)} dictionary.
  """
    tmp_bindings = defaultdict(set)
    for binding in bindings:
        tmp_bindings[binding['role']].update(binding['members'])
    return tmp_bindings