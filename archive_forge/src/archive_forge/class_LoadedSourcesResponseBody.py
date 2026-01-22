from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class LoadedSourcesResponseBody(BaseSchema):
    """
    "body" of LoadedSourcesResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'sources': {'type': 'array', 'items': {'$ref': '#/definitions/Source'}, 'description': 'Set of loaded sources.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, sources, update_ids_from_dap=False, **kwargs):
        """
        :param array sources: Set of loaded sources.
        """
        self.sources = sources
        if update_ids_from_dap and self.sources:
            for o in self.sources:
                Source.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        sources = self.sources
        if sources and hasattr(sources[0], 'to_dict'):
            sources = [x.to_dict() for x in sources]
        dct = {'sources': [Source.update_dict_ids_to_dap(o) for o in sources] if update_ids_to_dap and sources else sources}
        dct.update(self.kwargs)
        return dct