from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('setBreakpoints')
@register
class SetBreakpointsRequest(BaseSchema):
    """
    Sets multiple breakpoints for a single source and clears all previous breakpoints in that source.
    
    To clear all breakpoint for a source, specify an empty array.
    
    When a breakpoint is hit, a 'stopped' event (with reason 'breakpoint') is generated.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['setBreakpoints']}, 'arguments': {'type': 'SetBreakpointsArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param SetBreakpointsArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'setBreakpoints'
        if arguments is None:
            self.arguments = SetBreakpointsArguments()
        else:
            self.arguments = SetBreakpointsArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != SetBreakpointsArguments else arguments
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