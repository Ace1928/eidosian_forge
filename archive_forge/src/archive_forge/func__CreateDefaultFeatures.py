import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _CreateDefaultFeatures(self, edition):
    """Creates a FeatureSet message with defaults for a specific edition.

    Args:
      edition: the edition to generate defaults for.

    Returns:
      A FeatureSet message with defaults for a specific edition.
    """
    from google.protobuf import descriptor_pb2
    with _edition_defaults_lock:
        if not self._edition_defaults:
            self._edition_defaults = descriptor_pb2.FeatureSetDefaults()
            self._edition_defaults.ParseFromString(self._serialized_edition_defaults)
    if edition < self._edition_defaults.minimum_edition:
        raise TypeError('Edition %s is earlier than the minimum supported edition %s!' % (descriptor_pb2.Edition.Name(edition), descriptor_pb2.Edition.Name(self._edition_defaults.minimum_edition)))
    if edition > self._edition_defaults.maximum_edition:
        raise TypeError('Edition %s is later than the maximum supported edition %s!' % (descriptor_pb2.Edition.Name(edition), descriptor_pb2.Edition.Name(self._edition_defaults.maximum_edition)))
    found = None
    for d in self._edition_defaults.defaults:
        if d.edition > edition:
            break
        found = d.features
    if found is None:
        raise TypeError('No valid default found for edition %s!' % descriptor_pb2.Edition.Name(edition))
    defaults = descriptor_pb2.FeatureSet()
    defaults.CopyFrom(found)
    return defaults