from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('terminated')
@register
class TerminatedEvent(BaseSchema):
    """
    The event indicates that debugging of the debuggee has terminated. This does **not** mean that the
    debuggee itself has exited.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['terminated']}, 'body': {'type': 'object', 'properties': {'restart': {'type': ['array', 'boolean', 'integer', 'null', 'number', 'object', 'string'], 'description': "A debug adapter may set 'restart' to true (or to an arbitrary object) to request that the front end restarts the session.\nThe value is not interpreted by the client and passed unmodified as an attribute '__restart' to the 'launch' and 'attach' requests."}}}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, seq=-1, body=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param TerminatedEventBody body: 
        """
        self.type = 'event'
        self.event = 'terminated'
        self.seq = seq
        if body is None:
            self.body = TerminatedEventBody()
        else:
            self.body = TerminatedEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != TerminatedEventBody else body
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        event = self.event
        seq = self.seq
        body = self.body
        dct = {'type': type, 'event': event, 'seq': seq}
        if body is not None:
            dct['body'] = body.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct