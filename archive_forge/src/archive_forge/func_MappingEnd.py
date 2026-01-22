from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def MappingEnd(self, event, loader):
    """End of mapping

    Args:
      event: Ignored.
      loader: Ignored.
    """
    assert self._top[0] == _TOKEN_MAPPING
    end_object = self._Pop()
    top_value = self._top[1]
    self._builder.EndMapping(top_value, end_object)