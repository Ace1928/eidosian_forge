from oslo_serialization import jsonutils
from oslo_serialization.serializer.base_serializer import BaseSerializer
class JSONSerializer(BaseSerializer):
    """JSON serializer based on the jsonutils module."""

    def __init__(self, default=jsonutils.to_primitive, encoding='utf-8'):
        self._default = default
        self._encoding = encoding

    def dump(self, obj, fp):
        return jsonutils.dump(obj, fp)

    def dump_as_bytes(self, obj):
        return jsonutils.dump_as_bytes(obj, default=self._default, encoding=self._encoding)

    def load(self, fp):
        return jsonutils.load(fp, encoding=self._encoding)

    def load_from_bytes(self, s):
        return jsonutils.loads(s, encoding=self._encoding)