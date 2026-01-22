from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ExceptionBreakMode(BaseSchema):
    """
    This enumeration defines all possible conditions when a thrown exception should result in a break.
    
    never: never breaks,
    
    always: always breaks,
    
    unhandled: breaks when exception unhandled,
    
    userUnhandled: breaks if the exception is not handled by user code.

    Note: automatically generated code. Do not edit manually.
    """
    NEVER = 'never'
    ALWAYS = 'always'
    UNHANDLED = 'unhandled'
    USERUNHANDLED = 'userUnhandled'
    VALID_VALUES = set(['never', 'always', 'unhandled', 'userUnhandled'])
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