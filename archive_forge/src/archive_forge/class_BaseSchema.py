from _pydevd_bundle._debug_adapter.pydevd_schema_log import debug_exception
import json
import itertools
from functools import partial
class BaseSchema(object):

    @staticmethod
    def initialize_ids_translation():
        BaseSchema._dap_id_to_obj_id = {0: 0, None: None}
        BaseSchema._obj_id_to_dap_id = {0: 0, None: None}
        BaseSchema._next_dap_id = partial(next, itertools.count(1))

    def to_json(self):
        return json.dumps(self.to_dict())

    @staticmethod
    def _translate_id_to_dap(obj_id):
        if obj_id == '*':
            return '*'
        dap_id = BaseSchema._obj_id_to_dap_id.get(obj_id)
        if dap_id is None:
            dap_id = BaseSchema._obj_id_to_dap_id[obj_id] = BaseSchema._next_dap_id()
            BaseSchema._dap_id_to_obj_id[dap_id] = obj_id
        return dap_id

    @staticmethod
    def _translate_id_from_dap(dap_id):
        if dap_id == '*':
            return '*'
        try:
            return BaseSchema._dap_id_to_obj_id[dap_id]
        except:
            raise KeyError('Wrong ID sent from the client: %s' % (dap_id,))

    @staticmethod
    def update_dict_ids_to_dap(dct):
        return dct

    @staticmethod
    def update_dict_ids_from_dap(dct):
        return dct