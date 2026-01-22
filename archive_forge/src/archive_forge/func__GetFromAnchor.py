from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _GetFromAnchor(self, anchor_value):
    try:
        resource_ref = self._resources.Parse(anchor_value, collection=self.collection_info.full_name)
    except resources.Error:
        return None
    except AttributeError:
        return None
    return getattr(resource_ref, self.parameter_name, None)