from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('breakpointLocations')
@register
class BreakpointLocationsRequest(BaseSchema):
    """
    The 'breakpointLocations' request returns all possible locations for source breakpoints in a given
    range.
    
    Clients should only call this request if the capability 'supportsBreakpointLocationsRequest' is
    true.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['breakpointLocations']}, 'arguments': {'type': 'BreakpointLocationsArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, seq=-1, arguments=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param BreakpointLocationsArguments arguments: 
        """
        self.type = 'request'
        self.command = 'breakpointLocations'
        self.seq = seq
        if arguments is None:
            self.arguments = BreakpointLocationsArguments()
        else:
            self.arguments = BreakpointLocationsArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != BreakpointLocationsArguments else arguments
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        command = self.command
        seq = self.seq
        arguments = self.arguments
        dct = {'type': type, 'command': command, 'seq': seq}
        if arguments is not None:
            dct['arguments'] = arguments.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct