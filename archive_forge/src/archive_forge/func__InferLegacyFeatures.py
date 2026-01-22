import abc
import binascii
import os
import threading
import warnings
from google.protobuf.internal import api_implementation
def _InferLegacyFeatures(self, edition, options, features):
    from google.protobuf import descriptor_pb2
    if edition >= descriptor_pb2.Edition.EDITION_2023:
        return
    if self._label == FieldDescriptor.LABEL_REQUIRED:
        features.field_presence = descriptor_pb2.FeatureSet.FieldPresence.LEGACY_REQUIRED
    if self._type == FieldDescriptor.TYPE_GROUP:
        features.message_encoding = descriptor_pb2.FeatureSet.MessageEncoding.DELIMITED
    if options.HasField('packed'):
        features.repeated_field_encoding = descriptor_pb2.FeatureSet.RepeatedFieldEncoding.PACKED if options.packed else descriptor_pb2.FeatureSet.RepeatedFieldEncoding.EXPANDED