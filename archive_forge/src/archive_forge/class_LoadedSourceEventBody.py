from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class LoadedSourceEventBody(BaseSchema):
    """
    "body" of LoadedSourceEvent

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'reason': {'type': 'string', 'description': 'The reason for the event.', 'enum': ['new', 'changed', 'removed']}, 'source': {'description': 'The new, changed, or removed source.', 'type': 'Source'}}
    __refs__ = set(['source'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, reason, source, update_ids_from_dap=False, **kwargs):
        """
        :param string reason: The reason for the event.
        :param Source source: The new, changed, or removed source.
        """
        self.reason = reason
        if source is None:
            self.source = Source()
        else:
            self.source = Source(update_ids_from_dap=update_ids_from_dap, **source) if source.__class__ != Source else source
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        reason = self.reason
        source = self.source
        dct = {'reason': reason, 'source': source.to_dict(update_ids_to_dap=update_ids_to_dap)}
        dct.update(self.kwargs)
        return dct