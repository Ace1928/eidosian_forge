import collections
import threading
import warnings
from google.protobuf import descriptor
from google.protobuf import descriptor_database
from google.protobuf import text_encoding
from google.protobuf.internal import python_edition_defaults
from google.protobuf.internal import python_message
def _ConvertFileProtoToFileDescriptor(self, file_proto):
    """Creates a FileDescriptor from a proto or returns a cached copy.

    This method also has the side effect of loading all the symbols found in
    the file into the appropriate dictionaries in the pool.

    Args:
      file_proto: The proto to convert.

    Returns:
      A FileDescriptor matching the passed in proto.
    """
    if file_proto.name not in self._file_descriptors:
        built_deps = list(self._GetDeps(file_proto.dependency))
        direct_deps = [self.FindFileByName(n) for n in file_proto.dependency]
        public_deps = [direct_deps[i] for i in file_proto.public_dependency]
        from google.protobuf import descriptor_pb2
        file_descriptor = descriptor.FileDescriptor(pool=self, name=file_proto.name, package=file_proto.package, syntax=file_proto.syntax, edition=descriptor_pb2.Edition.Name(file_proto.edition), options=_OptionsOrNone(file_proto), serialized_pb=file_proto.SerializeToString(), dependencies=direct_deps, public_dependencies=public_deps, create_key=descriptor._internal_create_key)
        scope = {}
        for dependency in built_deps:
            scope.update(self._ExtractSymbols(dependency.message_types_by_name.values()))
            scope.update(((_PrefixWithDot(enum.full_name), enum) for enum in dependency.enum_types_by_name.values()))
        for message_type in file_proto.message_type:
            message_desc = self._ConvertMessageDescriptor(message_type, file_proto.package, file_descriptor, scope, file_proto.syntax)
            file_descriptor.message_types_by_name[message_desc.name] = message_desc
        for enum_type in file_proto.enum_type:
            file_descriptor.enum_types_by_name[enum_type.name] = self._ConvertEnumDescriptor(enum_type, file_proto.package, file_descriptor, None, scope, True)
        for index, extension_proto in enumerate(file_proto.extension):
            extension_desc = self._MakeFieldDescriptor(extension_proto, file_proto.package, index, file_descriptor, is_extension=True)
            extension_desc.containing_type = self._GetTypeFromScope(file_descriptor.package, extension_proto.extendee, scope)
            self._SetFieldType(extension_proto, extension_desc, file_descriptor.package, scope)
            file_descriptor.extensions_by_name[extension_desc.name] = extension_desc
        for desc_proto in file_proto.message_type:
            self._SetAllFieldTypes(file_proto.package, desc_proto, scope)
        if file_proto.package:
            desc_proto_prefix = _PrefixWithDot(file_proto.package)
        else:
            desc_proto_prefix = ''
        for desc_proto in file_proto.message_type:
            desc = self._GetTypeFromScope(desc_proto_prefix, desc_proto.name, scope)
            file_descriptor.message_types_by_name[desc_proto.name] = desc
        for index, service_proto in enumerate(file_proto.service):
            file_descriptor.services_by_name[service_proto.name] = self._MakeServiceDescriptor(service_proto, index, scope, file_proto.package, file_descriptor)
        self._file_descriptors[file_proto.name] = file_descriptor

    def AddExtensionForNested(message_type):
        for nested in message_type.nested_types:
            AddExtensionForNested(nested)
        for extension in message_type.extensions:
            self._AddExtensionDescriptor(extension)
    file_desc = self._file_descriptors[file_proto.name]
    for extension in file_desc.extensions_by_name.values():
        self._AddExtensionDescriptor(extension)
    for message_type in file_desc.message_types_by_name.values():
        AddExtensionForNested(message_type)
    return file_desc