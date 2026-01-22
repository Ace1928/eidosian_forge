from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class PydevdInfo(BaseSchema):
    """
    This object contains details on pydevd.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'usingCython': {'type': 'boolean', 'description': 'Specifies whether the cython native module is being used.'}, 'usingFrameEval': {'type': 'boolean', 'description': 'Specifies whether the frame eval native module is being used.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, usingCython=None, usingFrameEval=None, update_ids_from_dap=False, **kwargs):
        """
        :param boolean usingCython: Specifies whether the cython native module is being used.
        :param boolean usingFrameEval: Specifies whether the frame eval native module is being used.
        """
        self.usingCython = usingCython
        self.usingFrameEval = usingFrameEval
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        usingCython = self.usingCython
        usingFrameEval = self.usingFrameEval
        dct = {}
        if usingCython is not None:
            dct['usingCython'] = usingCython
        if usingFrameEval is not None:
            dct['usingFrameEval'] = usingFrameEval
        dct.update(self.kwargs)
        return dct