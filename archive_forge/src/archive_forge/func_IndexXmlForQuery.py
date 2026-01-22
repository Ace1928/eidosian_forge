from __future__ import absolute_import
from ruamel import yaml
import copy
import itertools
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine.api import appinfo
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.api import validation
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_object
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
def IndexXmlForQuery(kind, ancestor, props):
    """Return the composite index definition XML needed for a query.

  Given a query, the arguments for this method can be computed with:
    _, kind, ancestor, props = datastore_index.CompositeIndexForQuery(query)
    props = datastore_index.GetRecommendedIndexProperties(props)

  Args:
    kind: the kind or None
    ancestor: True if this is an ancestor query, False otherwise
    props: PropertySpec objects

  Returns:
    A string with the XML for the composite index needed by the query.
  """
    serialized_xml = []
    is_geo = any((p.mode is GEOSPATIAL for p in props))
    if is_geo:
        ancestor_clause = ''
    else:
        ancestor_clause = 'ancestor="%s"' % ('true' if ancestor else 'false',)
    serialized_xml.append('  <datastore-index kind="%s" %s>' % (kind, ancestor_clause))
    for prop in props:
        if prop.mode is GEOSPATIAL:
            qual = ' mode="geospatial"'
        elif is_geo:
            qual = ''
        else:
            qual = ' direction="%s"' % ('desc' if prop.direction == DESCENDING else 'asc')
        serialized_xml.append('    <property name="%s"%s />' % (prop.name, qual))
    serialized_xml.append('  </datastore-index>')
    return '\n'.join(serialized_xml)