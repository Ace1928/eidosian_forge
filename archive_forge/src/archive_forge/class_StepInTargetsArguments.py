from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class StepInTargetsArguments(BaseSchema):
    """
    Arguments for 'stepInTargets' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'frameId': {'type': 'integer', 'description': 'The stack frame for which to retrieve the possible stepIn targets.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, frameId, update_ids_from_dap=False, **kwargs):
        """
        :param integer frameId: The stack frame for which to retrieve the possible stepIn targets.
        """
        self.frameId = frameId
        if update_ids_from_dap:
            self.frameId = self._translate_id_from_dap(self.frameId)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'frameId' in dct:
            dct['frameId'] = cls._translate_id_from_dap(dct['frameId'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        frameId = self.frameId
        if update_ids_to_dap:
            if frameId is not None:
                frameId = self._translate_id_to_dap(frameId)
        dct = {'frameId': frameId}
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'frameId' in dct:
            dct['frameId'] = cls._translate_id_to_dap(dct['frameId'])
        return dct