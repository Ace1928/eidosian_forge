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
def _PropertiesToXml(self, properties):
    """ Returns a list of the XML representations of each of the given
    properties. Ignores properties that don't exist in this entity.

    Arg:
      properties: string or list of strings

    Returns:
      list of strings
    """
    xml_properties = []
    for propname in properties:
        if not self.has_key(propname):
            continue
        propname_xml = saxutils.quoteattr(propname)
        values = self[propname]
        if isinstance(values, list) and (not values):
            continue
        if not isinstance(values, list):
            values = [values]
        proptype = datastore_types.PropertyTypeName(values[0])
        proptype_xml = saxutils.quoteattr(proptype)
        escaped_values = self._XmlEscapeValues(propname)
        open_tag = u'<property name=%s type=%s>' % (propname_xml, proptype_xml)
        close_tag = u'</property>'
        xml_properties += [open_tag + val + close_tag for val in escaped_values]
    return xml_properties