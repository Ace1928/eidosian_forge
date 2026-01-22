from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('exited')
@register
class ExitedEvent(BaseSchema):
    """
    The event indicates that the debuggee has exited and returns its exit code.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['exited']}, 'body': {'type': 'object', 'properties': {'exitCode': {'type': 'integer', 'description': 'The exit code returned from the debuggee.'}}, 'required': ['exitCode']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param ExitedEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'exited'
        if body is None:
            self.body = ExitedEventBody()
        else:
            self.body = ExitedEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != ExitedEventBody else body
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