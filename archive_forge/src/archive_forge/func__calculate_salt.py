import collections
import inspect
import logging
from cloudsdk.google.protobuf import descriptor_pb2
from cloudsdk.google.protobuf import descriptor_pool
from cloudsdk.google.protobuf import message
from cloudsdk.google.protobuf import reflection
from proto.marshal.rules.message import MessageRule
def _calculate_salt(self, new_class, fallback):
    manifest = self._get_manifest(new_class)
    if manifest and new_class.__name__ not in manifest:
        log.warning('proto-plus module {module} has a declared manifest but {class_name} is not in it'.format(module=inspect.getmodule(new_class).__name__, class_name=new_class.__name__))
    return '' if new_class.__name__ in manifest else (fallback or '').lower()