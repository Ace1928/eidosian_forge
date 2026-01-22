from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register_response('setExceptionBreakpoints')
@register
class SetExceptionBreakpointsResponse(BaseSchema):
    """
    Response to 'setExceptionBreakpoints' request.
    
    The response contains an array of Breakpoint objects with information about each exception
    breakpoint or filter. The Breakpoint objects are in the same order as the elements of the 'filters',
    'filterOptions', 'exceptionOptions' arrays given as arguments. If both 'filters' and 'filterOptions'
    are given, the returned array must start with 'filters' information first, followed by
    'filterOptions' information.
    
    The mandatory 'verified' property of a Breakpoint object signals whether the exception breakpoint or
    filter could be successfully created and whether the optional condition or hit count expressions are
    valid. In case of an error the 'message' property explains the problem. An optional 'id' property
    can be used to introduce a unique ID for the exception breakpoint or filter so that it can be
    updated subsequently by sending breakpoint events.
    
    For backward compatibility both the 'breakpoints' array and the enclosing 'body' are optional. If
    these elements are missing a client will not be able to show problems for individual exception
    breakpoints or filters.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'seq': {'type': 'integer', 'description': "Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request."}, 'type': {'type': 'string', 'enum': ['response']}, 'request_seq': {'type': 'integer', 'description': 'Sequence number of the corresponding request.'}, 'success': {'type': 'boolean', 'description': "Outcome of the request.\nIf true, the request was successful and the 'body' attribute may contain the result of the request.\nIf the value is false, the attribute 'message' contains the error in short form and the 'body' may contain additional information (see 'ErrorResponse.body.error')."}, 'command': {'type': 'string', 'description': 'The command requested.'}, 'message': {'type': 'string', 'description': "Contains the raw error in short form if 'success' is false.\nThis raw error might be interpreted by the frontend and is not shown in the UI.\nSome predefined values exist.", '_enum': ['cancelled'], 'enumDescriptions': ['request was cancelled.']}, 'body': {'type': 'object', 'properties': {'breakpoints': {'type': 'array', 'items': {'$ref': '#/definitions/Breakpoint'}, 'description': "Information about the exception breakpoints or filters.\nThe breakpoints returned are in the same order as the elements of the 'filters', 'filterOptions', 'exceptionOptions' arrays in the arguments. If both 'filters' and 'filterOptions' are given, the returned array must start with 'filters' information first, followed by 'filterOptions' information."}}}}
    __refs__ = set(['body'])
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, request_seq, success, command, seq=-1, message=None, body=None, update_ids_from_dap=False, **kwargs):
        """
        :param string type: 
        :param integer request_seq: Sequence number of the corresponding request.
        :param boolean success: Outcome of the request.
        If true, the request was successful and the 'body' attribute may contain the result of the request.
        If the value is false, the attribute 'message' contains the error in short form and the 'body' may contain additional information (see 'ErrorResponse.body.error').
        :param string command: The command requested.
        :param integer seq: Sequence number (also known as message ID). For protocol messages of type 'request' this ID can be used to cancel the request.
        :param string message: Contains the raw error in short form if 'success' is false.
        This raw error might be interpreted by the frontend and is not shown in the UI.
        Some predefined values exist.
        :param SetExceptionBreakpointsResponseBody body: 
        """
        self.type = 'response'
        self.request_seq = request_seq
        self.success = success
        self.command = command
        self.seq = seq
        self.message = message
        if body is None:
            self.body = SetExceptionBreakpointsResponseBody()
        else:
            self.body = SetExceptionBreakpointsResponseBody(update_ids_from_dap=update_ids_from_dap, **body) if body.__class__ != SetExceptionBreakpointsResponseBody else body
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        type = self.type
        request_seq = self.request_seq
        success = self.success
        command = self.command
        seq = self.seq
        message = self.message
        body = self.body
        dct = {'type': type, 'request_seq': request_seq, 'success': success, 'command': command, 'seq': seq}
        if message is not None:
            dct['message'] = message
        if body is not None:
            dct['body'] = body.to_dict(update_ids_to_dap=update_ids_to_dap)
        dct.update(self.kwargs)
        return dct