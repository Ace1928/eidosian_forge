import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def SetFeatureSetDefaults(self, defaults):
    """Sets the default feature mappings used during the build.

    Args:
      defaults: a FeatureSetDefaults message containing the new mappings.
    """
    if self._edition_defaults is not None:
        raise ValueError("Feature set defaults can't be changed once the pool has started building!")
    from google.protobuf import descriptor_pb2
    if not isinstance(defaults, descriptor_pb2.FeatureSetDefaults):
        raise TypeError('SetFeatureSetDefaults called with invalid type')
    if defaults.minimum_edition > defaults.maximum_edition:
        raise ValueError('Invalid edition range %s to %s' % (descriptor_pb2.Edition.Name(defaults.minimum_edition), descriptor_pb2.Edition.Name(defaults.maximum_edition)))
    prev_edition = descriptor_pb2.Edition.EDITION_UNKNOWN
    for d in defaults.defaults:
        if d.edition == descriptor_pb2.Edition.EDITION_UNKNOWN:
            raise ValueError('Invalid edition EDITION_UNKNOWN specified')
        if prev_edition >= d.edition:
            raise ValueError('Feature set defaults are not strictly increasing.  %s is greater than or equal to %s' % (descriptor_pb2.Edition.Name(prev_edition), descriptor_pb2.Edition.Name(d.edition)))
        prev_edition = d.edition
    self._edition_defaults = defaults