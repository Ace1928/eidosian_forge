import heapq
import itertools
import logging
import os
import re
import sys
import threading  # Knowing full well that this is a usually a placeholder.
import traceback
from xml.sax import saxutils
from googlecloudsdk.core.util import encoding
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import capabilities
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_query
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
def _XmlEscapeValues(self, property):
    """ Returns a list of the XML-escaped string values for the given property.
    Raises an AssertionError if the property doesn't exist.

    Arg:
      property: string

    Returns:
      list of strings
    """
    assert self.has_key(property)
    xml = []
    values = self[property]
    if not isinstance(values, list):
        values = [values]
    for val in values:
        if hasattr(val, 'ToXml'):
            xml.append(val.ToXml())
        elif val is None:
            xml.append('')
        else:
            xml.append(saxutils.escape(unicode(val)))
    return xml