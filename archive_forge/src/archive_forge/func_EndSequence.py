from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def EndSequence(self, top_value, sequence):
    """Previously constructed sequence scope is at an end.

    Called when the end of a sequence block is encountered.  Useful for
    additional clean up or end of scope validation.

    Args:
      top_value: Value which is parent of the sequence.
      sequence: Sequence which is at the end of its scope.
    """