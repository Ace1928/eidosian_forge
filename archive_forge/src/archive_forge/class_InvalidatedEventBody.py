from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class InvalidatedEventBody(BaseSchema):
    """
    "body" of InvalidatedEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'areas': {'type': 'array', 'description': "Optional set of logical areas that got invalidated. This property has a hint characteristic: a client can only be expected to make a 'best effort' in honouring the areas but there are no guarantees. If this property is missing, empty, or if values are not understand the client should assume a single value 'all'.", 'items': {'$ref': '#/definitions/InvalidatedAreas'}}, 'threadId': {'type': 'integer', 'description': 'If specified, the client only needs to refetch data related to this thread.'}, 'stackFrameId': {'type': 'integer', 'description': "If specified, the client only needs to refetch data related to this stack frame (and the 'threadId' is ignored)."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, areas=None, threadId=None, stackFrameId=None, update_ids_from_dap=False, **kwargs):
        """
        :param array areas: Optional set of logical areas that got invalidated. This property has a hint characteristic: a client can only be expected to make a 'best effort' in honouring the areas but there are no guarantees. If this property is missing, empty, or if values are not understand the client should assume a single value 'all'.
        :param integer threadId: If specified, the client only needs to refetch data related to this thread.
        :param integer stackFrameId: If specified, the client only needs to refetch data related to this stack frame (and the 'threadId' is ignored).
        """
        self.areas = areas
        if update_ids_from_dap and self.areas:
            for o in self.areas:
                InvalidatedAreas.update_dict_ids_from_dap(o)
        self.threadId = threadId
        self.stackFrameId = stackFrameId
        if update_ids_from_dap:
            self.threadId = self._translate_id_from_dap(self.threadId)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_from_dap(dct['threadId'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        areas = self.areas
        if areas and hasattr(areas[0], 'to_dict'):
            areas = [x.to_dict() for x in areas]
        threadId = self.threadId
        stackFrameId = self.stackFrameId
        if update_ids_to_dap:
            if threadId is not None:
                threadId = self._translate_id_to_dap(threadId)
        dct = {}
        if areas is not None:
            dct['areas'] = [InvalidatedAreas.update_dict_ids_to_dap(o) for o in areas] if update_ids_to_dap and areas else areas
        if threadId is not None:
            dct['threadId'] = threadId
        if stackFrameId is not None:
            dct['stackFrameId'] = stackFrameId
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_to_dap(dct['threadId'])
        return dct