from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('setExceptionBreakpoints')
@register
class SetExceptionBreakpointsRequest(BaseSchema):
    """
    The request configures the debuggers response to thrown exceptions.
    
    If an exception is configured to break, a 'stopped' event is fired (with reason 'exception').
    
    Clients should only call this request if the capability 'exceptionBreakpointFilters' returns one or
    more filters.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['setExceptionBreakpoints']}, 'arguments': {'type': 'SetExceptionBreakpointsArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param SetExceptionBreakpointsArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'setExceptionBreakpoints'
        if arguments is None:
            self.arguments = SetExceptionBreakpointsArguments()
        else:
            self.arguments = SetExceptionBreakpointsArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != SetExceptionBreakpointsArguments else arguments
        self.seq = seq
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        command = self.command
        arguments = self.arguments
        seq = self.seq
        dct = {'type': type, 'command': command, 'arguments': arguments.to_dict(update_ids_to_dap=update_ids_to_dap), 'seq': seq}
        dct.update(self.kwargs)
        return dct