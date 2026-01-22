from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('invalidated')
@register
class InvalidatedEvent(BaseSchema):
    """
    This event signals that some state in the debug adapter has changed and requires that the client
    needs to re-render the data snapshot previously requested.
    
    Debug adapters do not have to emit this event for runtime changes like stopped or thread events
    because in that case the client refetches the new state anyway. But the event can be used for
    example to refresh the UI after rendering formatting has changed in the debug adapter.
    
    This event should only be sent if the debug adapter has received a value true for the
    'supportsInvalidatedEvent' capability of the 'initialize' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['invalidated']}, 'body': {'type': 'object', 'properties': {'areas': {'type': 'array', 'description': "Optional set of logical areas that got invalidated. This property has a hint characteristic: a client can only be expected to make a 'best effort' in honouring the areas but there are no guarantees. If this property is missing, empty, or if values are not understand the client should assume a single value 'all'.", 'items': {'$ref': '#/definitions/InvalidatedAreas'}}, 'threadId': {'type': 'integer', 'description': 'If specified, the client only needs to refetch data related to this thread.'}, 'stackFrameId': {'type': 'integer', 'description': "If specified, the client only needs to refetch data related to this stack frame (and the 'threadId' is ignored)."}}}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param InvalidatedEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'invalidated'
        if body is None:
            self.body = InvalidatedEventBody()
        else:
            self.body = InvalidatedEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != InvalidatedEventBody else body
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