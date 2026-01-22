import contextlib
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
class ApiDefMap(object):
    """Wrapper around Tf_ApiDefMap that handles querying and deletion.

  The OpDef protos are also stored in this class so that they could
  be queried by op name.
  """
    __slots__ = ['_api_def_map', '_op_per_name']

    def __init__(self):
        op_def_proto = op_def_pb2.OpList()
        buf = c_api.TF_GetAllOpList()
        try:
            op_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
            self._api_def_map = c_api.TF_NewApiDefMap(buf)
        finally:
            c_api.TF_DeleteBuffer(buf)
        self._op_per_name = {}
        for op in op_def_proto.op:
            self._op_per_name[op.name] = op

    def __del__(self):
        if c_api is not None and c_api.TF_DeleteApiDefMap is not None:
            c_api.TF_DeleteApiDefMap(self._api_def_map)

    def put_api_def(self, text):
        c_api.TF_ApiDefMapPut(self._api_def_map, text, len(text))

    def get_api_def(self, op_name):
        api_def_proto = api_def_pb2.ApiDef()
        buf = c_api.TF_ApiDefMapGet(self._api_def_map, op_name, len(op_name))
        try:
            api_def_proto.ParseFromString(c_api.TF_GetBuffer(buf))
        finally:
            c_api.TF_DeleteBuffer(buf)
        return api_def_proto

    def get_op_def(self, op_name):
        if op_name in self._op_per_name:
            return self._op_per_name[op_name]
        raise ValueError(f'No op_def found for op name {op_name}.')

    def op_names(self):
        return self._op_per_name.keys()