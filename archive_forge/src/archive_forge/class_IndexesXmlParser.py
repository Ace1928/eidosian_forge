from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from xml.etree import ElementTree
from googlecloudsdk.third_party.appengine.api.validation import ValidationError
from googlecloudsdk.third_party.appengine.datastore.datastore_index import Index
from googlecloudsdk.third_party.appengine.datastore.datastore_index import IndexDefinitions
from googlecloudsdk.third_party.appengine.datastore.datastore_index import Property
class IndexesXmlParser(object):
    """Provides logic for walking down XML tree and pulling data."""

    def Parse(self, xml_str):
        """Parses XML string and returns object representation of relevant info.

    Args:
      xml_str: The XML string.
    Returns:
      An IndexDefinitions object containing the result of parsing the XML.
    Raises:
      ValidationError: In case of malformed XML or illegal inputs.
    """
        try:
            self.indexes = []
            self.errors = []
            xml_root = ElementTree.fromstring(xml_str)
            if xml_root.tag != 'datastore-indexes':
                raise ValidationError('Root tag must be <datastore-indexes>')
            for child in list(xml_root):
                self.ProcessIndexNode(child)
            if self.errors:
                raise ValidationError('\n'.join(self.errors))
            return IndexDefinitions(indexes=self.indexes)
        except ElementTree.ParseError as e:
            raise ValidationError('Bad input -- not valid XML: %s' % e)

    def ProcessIndexNode(self, node):
        """Processes XML <datastore-index> nodes into Index objects.

    The following information is parsed out:
    kind: specifies the kind of entities to index.
    ancestor: true if the index supports queries that filter by
      ancestor-key to constraint results to a single entity group.
    property: represents the entity properties to index, with a name
      and direction attribute.

    Args:
      node: <datastore-index> XML node in datastore-indexes.xml.
    """
        if node.tag != 'datastore-index':
            self.errors.append('Unrecognized node: <%s>' % node.tag)
            return
        index = Index()
        index.kind = node.attrib.get('kind')
        if not index.kind:
            self.errors.append(MISSING_KIND)
        ancestor = node.attrib.get('ancestor', 'false')
        index.ancestor = _BooleanAttribute(ancestor)
        if index.ancestor is None:
            self.errors.append('Value for ancestor should be true or false, not "%s"' % ancestor)
        properties = []
        property_nodes = [n for n in list(node) if n.tag == 'property']
        has_geospatial = any((property_node.attrib.get('mode') == 'geospatial' for property_node in property_nodes))
        for property_node in property_nodes:
            name = property_node.attrib.get('name', '')
            if not name:
                self.errors.append(NAME_MISSING % index.kind)
                continue
            direction = property_node.attrib.get('direction')
            mode = property_node.attrib.get('mode')
            if mode:
                if index.ancestor:
                    self.errors.append(MODE_AND_ANCESTOR_SPECIFIED)
                    continue
                if mode != 'geospatial':
                    self.errors.append(BAD_MODE % mode)
                    continue
                if direction:
                    self.errors.append(MODE_AND_DIRECTION_SPECIFIED)
                    continue
            elif not direction:
                if not has_geospatial:
                    direction = 'asc'
            elif direction not in ('asc', 'desc'):
                self.errors.append(BAD_DIRECTION % direction)
                continue
            properties.append(Property(name=name, direction=direction, mode=mode))
        index.properties = properties
        self.indexes.append(index)