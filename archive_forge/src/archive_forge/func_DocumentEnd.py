from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def DocumentEnd(self, event, loader):
    """End of document.

    Args:
      event: Ignored.
    """
    assert self._top[0] == _TOKEN_DOCUMENT
    self._results.append(self._Pop())