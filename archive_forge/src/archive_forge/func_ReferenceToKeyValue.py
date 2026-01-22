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
def ReferenceToKeyValue(key, id_resolver=None):
    """Converts a key into a comparable hashable "key" value.

  Args:
    key: The entity_pb.Reference or googledatastore.Key from which to construct
        the key value.
    id_resolver: An optional datastore_pbs.IdResolver. Only necessary for
        googledatastore.Key values.
  Returns:
    A comparable and hashable representation of the given key that is
    compatible with one derived from a key property value.
  """
    if datastore_pbs._CLOUD_DATASTORE_ENABLED and isinstance(key, googledatastore.Key):
        v1_key = key
        key = entity_pb.Reference()
        datastore_pbs.get_entity_converter(id_resolver).v1_to_v3_reference(v1_key, key)
    elif isinstance(key, entity_v4_pb.Key):
        v4_key = key
        key = entity_pb.Reference()
        datastore_pbs.get_entity_converter().v4_to_v3_reference(v4_key, key)
    if isinstance(key, entity_pb.Reference):
        element_list = key.path().element_list()
    elif isinstance(key, entity_pb.PropertyValue_ReferenceValue):
        element_list = key.pathelement_list()
    else:
        raise datastore_errors.BadArgumentError('key arg expected to be entity_pb.Reference or googledatastore.Key (%r)' % (key,))
    result = [entity_pb.PropertyValue.kReferenceValueGroup, key.app(), key.name_space()]
    for element in element_list:
        result.append(element.type())
        if element.has_name():
            result.append(element.name())
        else:
            result.append(element.id())
    return tuple(result)