from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ScopesResponseBody(BaseSchema):
    """
    "body" of ScopesResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'scopes': {'type': 'array', 'items': {'$ref': '#/definitions/Scope'}, 'description': 'The scopes of the stackframe. If the array has length zero, there are no scopes available.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, scopes, update_ids_from_dap=False, **kwargs):
        """
        :param array scopes: The scopes of the stackframe. If the array has length zero, there are no scopes available.
        """
        self.scopes = scopes
        if update_ids_from_dap and self.scopes:
            for o in self.scopes:
                Scope.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        scopes = self.scopes
        if scopes and hasattr(scopes[0], 'to_dict'):
            scopes = [x.to_dict() for x in scopes]
        dct = {'scopes': [Scope.update_dict_ids_to_dap(o) for o in scopes] if update_ids_to_dap and scopes else scopes}
        dct.update(self.kwargs)
        return dct