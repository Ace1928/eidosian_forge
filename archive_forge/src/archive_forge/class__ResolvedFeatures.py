from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
class _ResolvedFeatures:

    def __init__(self, features=None, **kwargs):
        if features:
            for k, v in features.FIELDS.items():
                setattr(self, k, getattr(features, k))
        else:
            for k, v in kwargs.items():
                setattr(self, k, v)