from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def AppendTo(self, subject, value):
    """Append value to a sequence representation.

    Implementation is defined by sub-class of Builder.

    Args:
      subject: Object that represents sequence.  Value returned from
        BuildSequence
      value: Value to be appended to subject.  Can be any kind of value.
    """