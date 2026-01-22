from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GetConfigLocationProperty():
    """Returns the value of the metastore/location config property."""
    return properties.VALUES.metastore.location.GetOrFail()