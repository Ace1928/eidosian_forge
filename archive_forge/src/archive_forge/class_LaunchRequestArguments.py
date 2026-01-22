from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class LaunchRequestArguments(BaseSchema):
    """
    Arguments for 'launch' request. Additional attributes are implementation specific.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'noDebug': {'type': 'boolean', 'description': 'If noDebug is true the launch request should launch the program without enabling debugging.'}, '__restart': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': "Optional data from the previous, restarted session.\nThe data is sent as the 'restart' attribute of the 'terminated' event.\nThe client should leave the data intact."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, noDebug=None, __restart=None, update_ids_from_dap=False, **kwargs):
        """
        :param boolean noDebug: If noDebug is true the launch request should launch the program without enabling debugging.
        :param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] __restart: Optional data from the previous, restarted session.
        The data is sent as the 'restart' attribute of the 'terminated' event.
        The client should leave the data intact.
        """
        self.noDebug = noDebug
        self.__restart = __restart
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        noDebug = self.noDebug
        __restart = self.__restart
        dct = {}
        if noDebug is not None:
            dct['noDebug'] = noDebug
        if __restart is not None:
            dct['__restart'] = __restart
        dct.update(self.kwargs)
        return dct