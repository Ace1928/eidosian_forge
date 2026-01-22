from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdAuthorizeArguments(BaseSchema):
    """
    Arguments for 'pydevdAuthorize' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'debugServerAccessToken': {'type': 'string', 'description': 'The access token to access the debug server.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, debugServerAccessToken=None, update_ids_from_dap=False, **kwargs):
        """
        :param string debugServerAccessToken: The access token to access the debug server.
        """
        self.debugServerAccessToken = debugServerAccessToken
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        debugServerAccessToken = self.debugServerAccessToken
        dct = {}
        if debugServerAccessToken is not None:
            dct['debugServerAccessToken'] = debugServerAccessToken
        dct.update(self.kwargs)
        return dct