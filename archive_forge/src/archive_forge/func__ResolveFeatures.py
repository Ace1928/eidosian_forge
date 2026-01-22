import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _ResolveFeatures(self, edition, raw_options):
    """Resolves features from the raw options of this descriptor.

    Args:
      edition: The edition to use for feature defaults.
      raw_options: The options for this descriptor that are being processed.

    Returns:
      A fully resolved feature set for making runtime decisions.
    """
    from google.protobuf import descriptor_pb2
    if self._parent:
        features = descriptor_pb2.FeatureSet()
        features.CopyFrom(self._parent._GetFeatures())
    else:
        features = self.file.pool._CreateDefaultFeatures(edition)
    unresolved = descriptor_pb2.FeatureSet()
    unresolved.CopyFrom(raw_options.features)
    self._InferLegacyFeatures(edition, raw_options, unresolved)
    features.MergeFrom(unresolved)
    return self.file.pool._InternFeatures(features)