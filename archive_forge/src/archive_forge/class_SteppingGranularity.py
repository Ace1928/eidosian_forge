from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SteppingGranularity(BaseSchema):
    """
    The granularity of one 'step' in the stepping requests 'next', 'stepIn', 'stepOut', and 'stepBack'.

    Note: automatically generated code. Do not edit manually.
    """
    STATEMENT = 'statement'
    LINE = 'line'
    INSTRUCTION = 'instruction'
    VALID_VALUES = set(['statement', 'line', 'instruction'])
    __props__ = {}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, update_ids_from_dap=False, **kwargs):
        """
    
        """
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        dct = {}
        dct.update(self.kwargs)
        return dct