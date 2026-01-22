from . import xml6
class Serializer_v7(xml6.Serializer_v6):
    """A Serializer that supports tree references"""
    supported_kinds = {'file', 'directory', 'symlink', 'tree-reference'}
    format_num = b'7'