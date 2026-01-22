from __future__ import absolute_import
from ruamel import yaml
from googlecloudsdk.third_party.appengine.api import yaml_errors
from googlecloudsdk.third_party.appengine.api import yaml_listener
def MapTo(self, subject, key, value):
    """Map value to a mapping representation.

    Implementation is defined by sub-class of Builder.

    Args:
      subject: Object that represents mapping.  Value returned from
        BuildMapping.
      key: Key used to map value to subject.  Can be any scalar value.
      value: Value which is mapped to subject. Can be any kind of value.
    """