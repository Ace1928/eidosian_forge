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
def ToPropertyPb(name, values):
    """Creates type-specific entity_pb.PropertyValues.

  Determines the type and meaning of the PropertyValue based on the Python
  type of the input value(s).

  NOTE: This function does not validate anything!

  Args:
    name: string or unicode; the property name
    values: The values for this property, either a single one or a list of them.
      All values must be a supported type. Lists of values must all be of the
      same type.

  Returns:
    A list of entity_pb.Property instances.
  """
    encoded_name = name.encode('utf-8')
    values_type = type(values)
    if values_type is list and len(values) == 0:
        pb = entity_pb.Property()
        pb.set_meaning(entity_pb.Property.EMPTY_LIST)
        pb.set_name(encoded_name)
        pb.set_multiple(False)
        pb.mutable_value()
        return pb
    elif values_type is list:
        multiple = True
    else:
        multiple = False
        values = [values]
    pbs = []
    for v in values:
        pb = entity_pb.Property()
        pb.set_name(encoded_name)
        pb.set_multiple(multiple)
        meaning = _PROPERTY_MEANINGS.get(v.__class__)
        if meaning is not None:
            pb.set_meaning(meaning)
        if hasattr(v, 'meaning_uri') and v.meaning_uri:
            pb.set_meaning_uri(v.meaning_uri)
        pack_prop = _PACK_PROPERTY_VALUES[v.__class__]
        pbvalue = pack_prop(name, v, pb.mutable_value())
        pbs.append(pb)
    if multiple:
        return pbs
    else:
        return pbs[0]