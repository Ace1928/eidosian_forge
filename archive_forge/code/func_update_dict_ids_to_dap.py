from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@classmethod
def update_dict_ids_to_dap(cls, dct):
    if 'variablesReference' in dct:
        dct['variablesReference'] = cls._translate_id_to_dap(dct['variablesReference'])
    return dct