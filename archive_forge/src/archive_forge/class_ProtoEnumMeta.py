import enum
from cloudsdk.google.protobuf import descriptor_pb2
from proto import _file_info
from proto import _package_info
from proto.marshal.rules.enums import EnumRule
class ProtoEnumMeta(enum.EnumMeta):
    """A metaclass for building and registering protobuf enums."""

    def __new__(mcls, name, bases, attrs):
        if bases[0] == enum.IntEnum:
            return super().__new__(mcls, name, bases, attrs)
        package, marshal = _package_info.compile(name, attrs)
        local_path = tuple(attrs.get('__qualname__', name).split('.'))
        if '<locals>' in local_path:
            ix = local_path.index('<locals>')
            local_path = local_path[:ix - 1] + local_path[ix + 1:]
        full_name = '.'.join((package,) + local_path).lstrip('.')
        filename = _file_info._FileInfo.proto_file_name(attrs.get('__module__', name.lower()))
        pb_options = '_pb_options'
        opts = attrs.pop(pb_options, {})
        if pb_options in attrs._member_names:
            if isinstance(attrs._member_names, list):
                idx = attrs._member_names.index(pb_options)
                attrs._member_names.pop(idx)
            else:
                del attrs._member_names[pb_options]
        enum_desc = descriptor_pb2.EnumDescriptorProto(name=name, value=sorted((descriptor_pb2.EnumValueDescriptorProto(name=name, number=number) for name, number in attrs.items() if isinstance(number, int)), key=lambda v: v.number), options=opts)
        file_info = _file_info._FileInfo.maybe_add_descriptor(filename, package)
        if len(local_path) == 1:
            file_info.descriptor.enum_type.add().MergeFrom(enum_desc)
        else:
            file_info.nested_enum[local_path] = enum_desc
        cls = super().__new__(mcls, name, bases, attrs)
        cls._meta = _EnumInfo(full_name=full_name, pb=None)
        file_info.enums[full_name] = cls
        marshal.register(cls, EnumRule(cls))
        if file_info.ready(new_class=cls):
            file_info.generate_file_pb(new_class=cls, fallback_salt=full_name)
        return cls