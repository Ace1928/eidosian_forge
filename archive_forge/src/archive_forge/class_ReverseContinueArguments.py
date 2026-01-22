from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ReverseContinueArguments(BaseSchema):
    """
    Arguments for 'reverseContinue' request.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'threadId': {'type': 'integer', 'description': "Specifies the active thread. If the debug adapter supports single thread execution (see 'supportsSingleThreadExecutionRequests') and the optional argument 'singleThread' is true, only the thread with this ID is resumed."}, 'singleThread': {'type': 'boolean', 'description': "If this optional flag is true, backward execution is resumed only for the thread with given 'threadId'."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, threadId, singleThread=None, update_ids_from_dap=False, **kwargs):
        """
        :param integer threadId: Specifies the active thread. If the debug adapter supports single thread execution (see 'supportsSingleThreadExecutionRequests') and the optional argument 'singleThread' is true, only the thread with this ID is resumed.
        :param boolean singleThread: If this optional flag is true, backward execution is resumed only for the thread with given 'threadId'.
        """
        self.threadId = threadId
        self.singleThread = singleThread
        if update_ids_from_dap:
            self.threadId = self._translate_id_from_dap(self.threadId)
        self.kwargs = kwargs

    @classmethod
    def update_dict_ids_from_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_from_dap(dct['threadId'])
        return dct

    def to_dict(self, update_ids_to_dap=False):
        threadId = self.threadId
        singleThread = self.singleThread
        if update_ids_to_dap:
            if threadId is not None:
                threadId = self._translate_id_to_dap(threadId)
        dct = {'threadId': threadId}
        if singleThread is not None:
            dct['singleThread'] = singleThread
        dct.update(self.kwargs)
        return dct

    @classmethod
    def update_dict_ids_to_dap(cls, dct):
        if 'threadId' in dct:
            dct['threadId'] = cls._translate_id_to_dap(dct['threadId'])
        return dct