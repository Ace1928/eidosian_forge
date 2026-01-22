from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdAuthorizeResponseBody(BaseSchema):
    """
    "body" of PydevdAuthorizeResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'clientAccessToken': {'type': 'string', 'description': 'The access token to access the client (i.e.: usually the IDE).'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, clientAccessToken, update_ids_from_dap=False, **kwargs):
        """
        :param string clientAccessToken: The access token to access the client (i.e.: usually the IDE).
        """
        self.clientAccessToken = clientAccessToken
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        clientAccessToken = self.clientAccessToken
        dct = {'clientAccessToken': clientAccessToken}
        dct.update(self.kwargs)
        return dct