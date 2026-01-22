from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('initialized')
@register
class InitializedEvent(BaseSchema):
    """
    This event indicates that the debug adapter is ready to accept configuration requests (e.g.
    SetBreakpointsRequest, SetExceptionBreakpointsRequest).
    
    A debug adapter is expected to send this event when it is ready to accept configuration requests
    (but not before the 'initialize' request has finished).
    
    The sequence of events/requests is as follows:
    
    - adapters sends 'initialized' event (after the 'initialize' request has returned)
    
    - frontend sends zero or more 'setBreakpoints' requests
    
    - frontend sends one 'setFunctionBreakpoints' request (if capability 'supportsFunctionBreakpoints'
    is true)
    
    - frontend sends a 'setExceptionBreakpoints' request if one or more 'exceptionBreakpointFilters'
    have been defined (or if 'supportsConfigurationDoneRequest' is not defined or false)
    
    - frontend sends other future configuration requests
    
    - frontend sends one 'configurationDone' request to indicate the end of the configuration.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['initialized']}, 'body': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': 'Event-specific information.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, seq=-1, body=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'] body: Event-specific information.
        """
        self.type = 'event'
        self.event = 'initialized'
        self.seq = seq
        self.body = body
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        event = self.event
        seq = self.seq
        body = self.body
        dct = {'type': type, 'event': event, 'seq': seq}
        if body is not None:
            dct['body'] = body
        dct.update(self.kwargs)
        return dct