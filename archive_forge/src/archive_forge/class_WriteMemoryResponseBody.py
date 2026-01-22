from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class WriteMemoryResponseBody(BaseSchema):
    """
    "body" of WriteMemoryResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'offset': {'type': 'integer', 'description': "Optional property that should be returned when 'allowPartial' is true to indicate the offset of the first byte of data successfully written. Can be negative."}, 'bytesWritten': {'type': 'integer', 'description': "Optional property that should be returned when 'allowPartial' is true to indicate the number of bytes starting from address that were successfully written."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, offset=None, bytesWritten=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer offset: Optional property that should be returned when 'allowPartial' is true to indicate the offset of the first byte of data successfully written. Can be negative.
        :param integer bytesWritten: Optional property that should be returned when 'allowPartial' is true to indicate the number of bytes starting from address that were successfully written.
        """
        self.offset = offset
        self.bytesWritten = bytesWritten
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        offset = self.offset
        bytesWritten = self.bytesWritten
        dct = {}
        if offset is not None:
            dct['offset'] = offset
        if bytesWritten is not None:
            dct['bytesWritten'] = bytesWritten
        dct.update(self.kwargs)
        return dct