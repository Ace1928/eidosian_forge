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
def FromPropertyPb(pb):
    """Converts a property PB to a python value.

  Args:
    pb: entity_pb.Property

  Returns:
    # return type is determined by the type of the argument
    string, int, bool, double, users.User, or one of the atom or gd types
  """
    pbval = pb.value()
    meaning = pb.meaning()
    if pbval.has_stringvalue():
        value = pbval.stringvalue()
        if not pb.has_meaning() or meaning not in _NON_UTF8_MEANINGS:
            value = six_subset.text_type(value, 'utf-8')
    elif pbval.has_int64value():
        value = _PREFERRED_NUM_TYPE(pbval.int64value())
    elif pbval.has_booleanvalue():
        value = bool(pbval.booleanvalue())
    elif pbval.has_doublevalue():
        value = pbval.doublevalue()
    elif pbval.has_referencevalue():
        value = FromReferenceProperty(pbval)
    elif pbval.has_pointvalue():
        value = GeoPt(pbval.pointvalue().x(), pbval.pointvalue().y())
    elif pbval.has_uservalue():
        email = six_subset.text_type(pbval.uservalue().email(), 'utf-8')
        auth_domain = six_subset.text_type(pbval.uservalue().auth_domain(), 'utf-8')
        obfuscated_gaiaid = pbval.uservalue().obfuscated_gaiaid().decode('utf-8')
        obfuscated_gaiaid = six_subset.text_type(pbval.uservalue().obfuscated_gaiaid(), 'utf-8')
        federated_identity = None
        if pbval.uservalue().has_federated_identity():
            federated_identity = six_subset.text_type(pbval.uservalue().federated_identity(), 'utf-8')
        value = users.User(email=email, _auth_domain=auth_domain, _user_id=obfuscated_gaiaid, federated_identity=federated_identity, _strict_mode=False)
    else:
        value = None
    try:
        if pb.has_meaning() and meaning in _PROPERTY_CONVERSIONS:
            conversion = _PROPERTY_CONVERSIONS[meaning]
            value = conversion(value)
            if meaning == entity_pb.Property.BLOB and pb.has_meaning_uri():
                value.meaning_uri = pb.meaning_uri()
    except (KeyError, ValueError, IndexError, TypeError, AttributeError) as msg:
        raise datastore_errors.BadValueError('Error converting pb: %s\nException was: %s' % (pb, msg))
    return value