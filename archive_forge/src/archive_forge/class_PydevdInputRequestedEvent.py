from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('pydevdInputRequested')
@register
class PydevdInputRequestedEvent(BaseSchema):
    """
    The event indicates input was requested by debuggee.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['pydevdInputRequested']}, 'body': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': 'Event-specific information.'}}
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
        self.event = 'pydevdInputRequested'
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