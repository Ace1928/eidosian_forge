from __future__ import absolute_import
from __future__ import unicode_literals
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.api import datastore
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine._internal import six_subset
def _KindPropertiesToXml(self):
    """ Convert the properties that are part of this gd kind to XML. For
    testability, the XML elements in the output are sorted alphabetically
    by property name.

    Returns:
    string  # the XML representation of the gd kind properties
    """
    properties = self._kind_properties.intersection(set(self.keys()))
    xml = ''
    for prop in sorted(properties):
        prop_xml = saxutils.quoteattr(prop)[1:-1]
        value = self[prop]
        has_toxml = hasattr(value, 'ToXml') or (isinstance(value, list) and hasattr(value[0], 'ToXml'))
        for val in self._XmlEscapeValues(prop):
            if has_toxml:
                xml += '\n  %s' % val
            else:
                xml += '\n  <%s>%s</%s>' % (prop_xml, val, prop_xml)
    return xml