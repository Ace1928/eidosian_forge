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
def EncodeAppIdNamespace(app_id, namespace):
    """Concatenates app id and namespace into a single string.

  This method is needed for xml and datastore_file_stub.

  Args:
    app_id: The application id to encode
    namespace: The namespace to encode

  Returns:
    The string encoding for the app_id, namespace pair.
  """
    if not namespace:
        return app_id
    else:
        return app_id + _NAMESPACE_SEPARATOR + namespace