from __future__ import absolute_import
from __future__ import unicode_literals
from xml.sax import saxutils
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.api import datastore
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine._internal import six_subset
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