from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class SetExceptionBreakpointsResponseBody(BaseSchema):
    """
    "body" of SetExceptionBreakpointsResponse

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'breakpoints': {'type': 'array', 'items': {'$ref': '#/definitions/Breakpoint'}, 'description': "Information about the exception breakpoints or filters.\nThe breakpoints returned are in the same order as the elements of the 'filters', 'filterOptions', 'exceptionOptions' arrays in the arguments. If both 'filters' and 'filterOptions' are given, the returned array must start with 'filters' information first, followed by 'filterOptions' information."}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, breakpoints=None, update_ids_from_dap=False, **kwargs):
        """
        :param array breakpoints: Information about the exception breakpoints or filters.
        The breakpoints returned are in the same order as the elements of the 'filters', 'filterOptions', 'exceptionOptions' arrays in the arguments. If both 'filters' and 'filterOptions' are given, the returned array must start with 'filters' information first, followed by 'filterOptions' information.
        """
        self.breakpoints = breakpoints
        if update_ids_from_dap and self.breakpoints:
            for o in self.breakpoints:
                Breakpoint.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        breakpoints = self.breakpoints
        if breakpoints and hasattr(breakpoints[0], 'to_dict'):
            breakpoints = [x.to_dict() for x in breakpoints]
        dct = {}
        if breakpoints is not None:
            dct['breakpoints'] = [Breakpoint.update_dict_ids_to_dap(o) for o in breakpoints] if update_ids_to_dap and breakpoints else breakpoints
        dct.update(self.kwargs)
        return dct