from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_event('module')
@register
class ModuleEvent(BaseSchema):
    """
    The event indicates that some information about a module has changed.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['event']}, 'event': {'type': 'string', 'enum': ['module']}, 'body': {'type': 'object', 'properties': {'reason': {'type': 'string', 'description': 'The reason for the event.', 'enum': ['new', 'changed', 'removed']}, 'module': {'$ref': '#/definitions/Module', 'description': "The new, changed, or removed module. In case of 'removed' only the module id is used."}}, 'required': ['reason', 'module']}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, body, seq=-1, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param string event: 
        :param ModuleEventBody body: 
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        """
        self.type = 'event'
        self.event = 'module'
        if body is None:
            self.body = ModuleEventBody()
        else:
            self.body = ModuleEventBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != ModuleEventBody else body
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