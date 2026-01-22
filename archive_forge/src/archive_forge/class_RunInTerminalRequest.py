from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('runInTerminal')
@register
class RunInTerminalRequest(BaseSchema):
    """
    This optional request is sent from the debug adapter to the client to run a command in a terminal.
    
    This is typically used to launch the debuggee in a terminal provided by the client.
    
    This request should only be called if the client has passed the value true for the
    'supportsRunInTerminalRequest' capability of the 'initialize' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['runInTerminal']}, 'arguments': {'type': 'RunInTerminalRequestArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param RunInTerminalRequestArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'runInTerminal'
        if arguments is None:
            self.arguments = RunInTerminalRequestArguments()
        else:
            self.arguments = RunInTerminalRequestArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != RunInTerminalRequestArguments else arguments
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