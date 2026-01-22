from _pydevd_bundle._debug_adapter.pydevd_schema_log import debug_exception
import json
import itertools
from functools import partial
@staticmethod
def _translate_id_from_dap(dap_id):
    if dap_id == '*':
        return '*'
    try:
        return BaseSchema._dap_id_to_obj_id[dap_id]
    except:
        raise KeyError('Wrong ID sent from the client: %s' % (dap_id,))