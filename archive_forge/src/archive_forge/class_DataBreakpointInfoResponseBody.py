from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class DataBreakpointInfoResponseBody(BaseSchema):
    """
    "body" of DataBreakpointInfoResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'dataId': {'type': ['string', 'null'], 'description': 'An identifier for the data on which a data breakpoint can be registered with the setDataBreakpoints request or null if no data breakpoint is available.'}, 'description': {'type': 'string', 'description': 'UI string that describes on what data the breakpoint is set on or why a data breakpoint is not available.'}, 'accessTypes': {'type': 'array', 'items': {'$ref': '#/definitions/DataBreakpointAccessType'}, 'description': 'Optional attribute listing the available access types for a potential data breakpoint. A UI frontend could surface this information.'}, 'canPersist': {'type': 'boolean', 'description': 'Optional attribute indicating that a potential data breakpoint could be persisted across sessions.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, dataId, description, accessTypes=None, canPersist=None, update_ids_from_dap=False, **kwargs):
        """
        :param ['string', 'null'] dataId: An identifier for the data on which a data breakpoint can be registered with the setDataBreakpoints request or null if no data breakpoint is available.
        :param string description: UI string that describes on what data the breakpoint is set on or why a data breakpoint is not available.
        :param array accessTypes: Optional attribute listing the available access types for a potential data breakpoint. A UI frontend could surface this information.
        :param boolean canPersist: Optional attribute indicating that a potential data breakpoint could be persisted across sessions.
        """
        self.dataId = dataId
        self.description = description
        self.accessTypes = accessTypes
        if update_ids_from_dap and self.accessTypes:
            for o in self.accessTypes:
                DataBreakpointAccessType.update_dict_ids_from_dap(o)
        self.canPersist = canPersist
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        dataId = self.dataId
        description = self.description
        accessTypes = self.accessTypes
        if accessTypes and hasattr(accessTypes[0], 'to_dict'):
            accessTypes = [x.to_dict() for x in accessTypes]
        canPersist = self.canPersist
        dct = {'dataId': dataId, 'description': description}
        if accessTypes is not None:
            dct['accessTypes'] = [DataBreakpointAccessType.update_dict_ids_to_dap(o) for o in accessTypes] if update_ids_to_dap and accessTypes else accessTypes
        if canPersist is not None:
            dct['canPersist'] = canPersist
        dct.update(self.kwargs)
        return dct