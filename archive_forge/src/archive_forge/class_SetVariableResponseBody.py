from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SetVariableResponseBody(BaseSchema):
    """
    "body" of SetVariableResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'value': {'type': 'string', 'description': 'The new value of the variable.'}, 'type': {'type': 'string', 'description': 'The type of the new value. Typically shown in the UI when hovering over the value.'}, 'variablesReference': {'type': 'integer', 'description': 'If variablesReference is > 0, the new value is structured and its children can be retrieved by passing variablesReference to the VariablesRequest.\nThe value should be less than or equal to 2147483647 (2^31-1).'}, 'namedVariables': {'type': 'integer', 'description': 'The number of named child variables.\nThe client can use this optional information to present the variables in a paged UI and fetch them in chunks.\nThe value should be less than or equal to 2147483647 (2^31-1).'}, 'indexedVariables': {'type': 'integer', 'description': 'The number of indexed child variables.\nThe client can use this optional information to present the variables in a paged UI and fetch them in chunks.\nThe value should be less than or equal to 2147483647 (2^31-1).'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, value, type=None, variablesReference=None, namedVariables=None, indexedVariables=None, update_ids_from_dap=False, **kwargs):
        """
        :param string value: The new value of the variable.
        :param string type: The type of the new value. Typically shown in the UI when hovering over the value.
        :param integer variablesReference: If variablesReference is > 0, the new value is structured and its children can be retrieved by passing variablesReference to the VariablesRequest.
        The value should be less than or equal to 2147483647 (2^31-1).
        :param integer namedVariables: The number of named child variables.
        The client can use this optional information to present the variables in a paged UI and fetch them in chunks.
        The value should be less than or equal to 2147483647 (2^31-1).
        :param integer indexedVariables: The number of indexed child variables.
        The client can use this optional information to present the variables in a paged UI and fetch them in chunks.
        The value should be less than or equal to 2147483647 (2^31-1).
        """
        self.value = value
        self.type = type
        self.variablesReference = variablesReference
        self.namedVariables = namedVariables
        self.indexedVariables = indexedVariables
        if update_ids_from_dap:
            self.variablesReference = self._translate_id_from_dap(self.variablesReference)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_from_dap(dct['variablesReference'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        value = self.value
        type = self.type
        variablesReference = self.variablesReference
        namedVariables = self.namedVariables
        indexedVariables = self.indexedVariables
        if update_ids_to_dap:
            if variablesReference is not None:
                variablesReference = self._translate_id_to_dap(variablesReference)
        dct = {'value': value}
        if type is not None:
            dct['type'] = type
        if variablesReference is not None:
            dct['variablesReference'] = variablesReference
        if namedVariables is not None:
            dct['namedVariables'] = namedVariables
        if indexedVariables is not None:
            dct['indexedVariables'] = indexedVariables
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'variablesReference' in dct:
            dct['variablesReference'] = cls._translate_id_to_dap(dct['variablesReference'])
        return dct