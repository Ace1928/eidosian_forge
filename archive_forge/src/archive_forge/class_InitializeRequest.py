from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_request('initialize')
@register
class InitializeRequest(BaseSchema):
    """
    The 'initialize' request is sent as the first request from the client to the debug adapter
    
    in order to configure it with client capabilities and to retrieve capabilities from the debug
    adapter.
    
    Until the debug adapter has responded to with an 'initialize' response, the client must not send any
    additional requests or events to the debug adapter.
    
    In addition the debug adapter is not allowed to send any requests or events to the client until it
    has responded with an 'initialize' response.
    
    The 'initialize' request may only be sent once.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['request']}, 'command': {'type': 'string', 'enum': ['initialize']}, 'arguments': {'type': 'InitializeRequestArguments'}}
    __refs__ = set(['arguments'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, arguments, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string command: 
        :param InitializeRequestArguments arguments: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'request'
        self.command = 'initialize'
        if arguments is None:
            self.arguments = InitializeRequestArguments()
        else:
            self.arguments = InitializeRequestArguments(update_ids_from_dap=update_ids_from_dap, **arguments) if arguments.__class__ != InitializeRequestArguments else arguments
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