from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from googlecloudsdk.calliope.concepts import util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
class FullySpecifiedAnchorFallthrough(_FallthroughBase):
    """A fallthrough that gets a parameter from the value of the anchor."""

    def __init__(self, fallthroughs, collection_info, parameter_name, plural=False):
        """Initializes a fallthrough getting a parameter from the anchor.

    For anchor arguments which can be plural, returns the list.

    Args:
      fallthroughs: list[_FallthroughBase], any fallthrough for an anchor arg.
      collection_info: the info of the collection to parse the anchor as.
      parameter_name: str, the name of the parameter
      plural: bool, whether the expected result should be a list. Should be
        False for everything except the "anchor" arguments in a case where a
    """
        if plural:
            hint_suffix = 'with fully specified names'
        else:
            hint_suffix = 'with a fully specified name'
        hint = [f'{f.hint} {hint_suffix}' for f in fallthroughs]
        active = all((f.active for f in fallthroughs))
        super(FullySpecifiedAnchorFallthrough, self).__init__(hint, active=active, plural=plural)
        self.parameter_name = parameter_name
        self.collection_info = collection_info
        self._fallthroughs = tuple(fallthroughs)
        self._resources = resources.REGISTRY.Clone()
        self._resources.RegisterApiByName(self.collection_info.api_name, self.collection_info.api_version)

    def _GetFromAnchor(self, anchor_value):
        try:
            resource_ref = self._resources.Parse(anchor_value, collection=self.collection_info.full_name)
        except resources.Error:
            return None
        except AttributeError:
            return None
        return getattr(resource_ref, self.parameter_name, None)

    def _Call(self, parsed_args):
        try:
            anchor_value = GetFromFallthroughs(self._fallthroughs, parsed_args, attribute_name=self.parameter_name)
        except AttributeNotFoundError:
            return None
        return self._GetFromAnchor(anchor_value)

    def __eq__(self, other):
        return isinstance(other, self.__class__) and other._fallthroughs == self._fallthroughs and (other.collection_info == self.collection_info) and (other.parameter_name == self.parameter_name)

    def __hash__(self):
        return sum(map(hash, [self._fallthroughs, str(self.collection_info), self.parameter_name]))