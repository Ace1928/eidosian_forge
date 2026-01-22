from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('continued')
@register
class ContinuedEvent(BaseSchema):
    """
    The event indicates that the execution of the debuggee has continued.
    
    Please note: a debug adapter is not expected to send this event in response to a request that
    implies that execution continues, e.g. 'launch' or 'continue'.
    
    It is only necessary to send a 'continued' event if there was no previous request that implied this.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['continued']}, 'body': {'type': 'object', 'properties': {'threadId': {'type': 'integer', 'description': 'The thread which was continued.'}, 'allThreadsContinued': {'type': 'boolean', 'description': "If 'allThreadsContinued' is true, a debug adapter can announce that all threads have continued."}}, 'required': ['threadId']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param ContinuedEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'continued'
        if body is None:
            self.body = ContinuedEventBody()
        else:
            self.body = ContinuedEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != ContinuedEventBody else body
        self.seq = seq
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        event = self.event
        body = self.body
        seq = self.seq
        dct = {'type': type, 'event': event, 'body': body.to_dict(update_ids_to_dap=update_ids_to_dap), 'seq': seq}
        dct.update(self.kwargs)
        return dct