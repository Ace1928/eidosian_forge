from __future__ import absolute_import
from __future__ import unicode_literals
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.api import datastore
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine._internal import six_subset
class GdKind(datastore.Entity):
    """ A base class for gd namespace kinds.

  This class contains common logic for all gd namespace kinds. For example,
  this class translates datastore (app id, kind, key) tuples to tag:
  URIs appropriate for use in <key> tags.
  """
    HEADER = "<entry xmlns:gd='http://schemas.google.com/g/2005'>\n  <category scheme='http://schemas.google.com/g/2005#kind'\n            term='http://schemas.google.com/g/2005#%s' />"
    FOOTER = '\n</entry>'
    _kind_properties = set()
    _contact_properties = set()

    def __init__(self, kind, title, kind_properties, contact_properties=[]):
        """ Ctor.

    title is the name of this particular entity, e.g. Bob Jones or Mom's
    Birthday Party.

    kind_properties is a list of property names that should be included in
    this entity's XML encoding as first-class XML elements, instead of
    <property> elements. 'title' and 'content' are added to kind_properties
    automatically, and may not appear in contact_properties.

    contact_properties is a list of property names that are Keys that point to
    Contact entities, and should be included in this entity's XML encoding as
    <gd:who> elements. If a property name is included in both kind_properties
    and contact_properties, it is treated as a Contact property.

    Args:
    kind: string
    title: string
    kind_properties: list of strings
    contact_properties: list of string
    """
        datastore.Entity.__init__(self, kind)
        if not isinstance(title, six_subset.string_types):
            raise datastore_errors.BadValueError('Expected a string for title; received %s (a %s).' % (title, datastore_types.typename(title)))
        self['title'] = title
        self['content'] = ''
        self._contact_properties = set(contact_properties)
        assert not self._contact_properties.intersection(list(self.keys()))
        self._kind_properties = set(kind_properties) - self._contact_properties
        self._kind_properties.add('title')
        self._kind_properties.add('content')

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

    def _ContactPropertiesToXml(self):
        """ Convert this kind's Contact properties kind to XML. For testability,
    the XML elements in the output are sorted alphabetically by property name.

    Returns:
    string  # the XML representation of the Contact properties
    """
        properties = self._contact_properties.intersection(set(self.keys()))
        xml = ''
        for prop in sorted(properties):
            values = self[prop]
            if not isinstance(values, list):
                values = [values]
            for value in values:
                assert isinstance(value, datastore_types.Key)
                xml += '\n  <gd:who rel="http://schemas.google.com/g/2005#%s.%s>\n    <gd:entryLink href="%s" />\n  </gd:who>' % (self.kind().lower(), prop, value.ToTagUri())
        return xml

    def _LeftoverPropertiesToXml(self):
        """ Convert all of this entity's properties that *aren't* part of this gd
    kind to XML.

    Returns:
    string  # the XML representation of the leftover properties
    """
        leftovers = set(self.keys())
        leftovers -= self._kind_properties
        leftovers -= self._contact_properties
        if leftovers:
            return '\n  ' + '\n  '.join(self._PropertiesToXml(leftovers))
        else:
            return ''

    def ToXml(self):
        """ Returns an XML representation of this entity, as a string.
    """
        xml = GdKind.HEADER % self.kind().lower()
        xml += self._KindPropertiesToXml()
        xml += self._ContactPropertiesToXml()
        xml += self._LeftoverPropertiesToXml()
        xml += GdKind.FOOTER
        return xml