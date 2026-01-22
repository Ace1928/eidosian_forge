from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SourceResponseBody(BaseSchema):
    """
    "body" of SourceResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'content': {'type': 'string', 'description': 'Content of the source reference.'}, 'mimeType': {'type': 'string', 'description': 'Optional content type (mime type) of the source.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, content, mimeType=None, update_ids_from_dap=False, **kwargs):
        """
        :param string content: Content of the source reference.
        :param string mimeType: Optional content type (mime type) of the source.
        """
        self.content = content
        self.mimeType = mimeType
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        content = self.content
        mimeType = self.mimeType
        dct = {'content': content}
        if mimeType is not None:
            dct['mimeType'] = mimeType
        dct.update(self.kwargs)
        return dct