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
def ToXml(self):
    """Returns an XML representation of this entity. Atom and gd:namespace
    properties are converted to XML according to their respective schemas. For
    more information, see:

      http://www.atomenabled.org/developers/syndication/
      https://developers.google.com/gdata/docs/1.0/elements

    This is *not* optimized. It shouldn't be used anywhere near code that's
    performance-critical.
    """
    xml = u'<entity kind=%s' % saxutils.quoteattr(self.kind())
    if self.__key.has_id_or_name():
        xml += ' key=%s' % saxutils.quoteattr(str(self.__key))
    xml += '>'
    if self.__key.has_id_or_name():
        xml += '\n  <key>%s</key>' % self.__key.ToTagUri()
    properties = self.keys()
    if properties:
        properties.sort()
        xml += '\n  ' + '\n  '.join(self._PropertiesToXml(properties))
    xml += '\n</entity>\n'
    return xml