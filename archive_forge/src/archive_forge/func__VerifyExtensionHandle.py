from cloudsdk.google.protobuf.internal import type_checkers
from cloudsdk.google.protobuf.descriptor import FieldDescriptor
def _VerifyExtensionHandle(message, extension_handle):
    """Verify that the given extension handle is valid."""
    if not isinstance(extension_handle, FieldDescriptor):
        raise KeyError('HasExtension() expects an extension handle, got: %s' % extension_handle)
    if not extension_handle.is_extension:
        raise KeyError('"%s" is not an extension.' % extension_handle.full_name)
    if not extension_handle.containing_type:
        raise KeyError('"%s" is missing a containing_type.' % extension_handle.full_name)
    if extension_handle.containing_type is not message.DESCRIPTOR:
        raise KeyError('Extension "%s" extends message type "%s", but this message is of type "%s".' % (extension_handle.full_name, extension_handle.containing_type.full_name, message.DESCRIPTOR.full_name))