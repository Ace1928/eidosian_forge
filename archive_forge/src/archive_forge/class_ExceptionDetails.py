from .pydevd_base_schema import BaseSchema, register, register_request, register_response, register_event
@register
class ExceptionDetails(BaseSchema):
    """
    Detailed information about an exception that has occurred.

    Note: automatically generated code. Do not edit manually.
    """
    __props__ = {'message': {'type': 'string', 'description': 'Message contained in the exception.'}, 'typeName': {'type': 'string', 'description': 'Short type name of the exception object.'}, 'fullTypeName': {'type': 'string', 'description': 'Fully-qualified type name of the exception object.'}, 'evaluateName': {'type': 'string', 'description': 'Optional expression that can be evaluated in the current scope to obtain the exception object.'}, 'stackTrace': {'type': 'string', 'description': 'Stack trace at the time the exception was thrown.'}, 'innerException': {'type': 'array', 'items': {'$ref': '#/definitions/ExceptionDetails'}, 'description': 'Details of the exception contained by this exception, if any.'}}
    __refs__ = set()
    __slots__ = list(__props__.keys()) + ['kwargs']

    def __init__(self, message=None, typeName=None, fullTypeName=None, evaluateName=None, stackTrace=None, innerException=None, update_ids_from_dap=False, **kwargs):
        """
        :param string message: Message contained in the exception.
        :param string typeName: Short type name of the exception object.
        :param string fullTypeName: Fully-qualified type name of the exception object.
        :param string evaluateName: Optional expression that can be evaluated in the current scope to obtain the exception object.
        :param string stackTrace: Stack trace at the time the exception was thrown.
        :param array innerException: Details of the exception contained by this exception, if any.
        """
        self.message = message
        self.typeName = typeName
        self.fullTypeName = fullTypeName
        self.evaluateName = evaluateName
        self.stackTrace = stackTrace
        self.innerException = innerException
        if update_ids_from_dap and self.innerException:
            for o in self.innerException:
                ExceptionDetails.update_dict_ids_from_dap(o)
        self.kwargs = kwargs

    def to_dict(self, update_ids_to_dap=False):
        message = self.message
        typeName = self.typeName
        fullTypeName = self.fullTypeName
        evaluateName = self.evaluateName
        stackTrace = self.stackTrace
        innerException = self.innerException
        if innerException and hasattr(innerException[0], 'to_dict'):
            innerException = [x.to_dict() for x in innerException]
        dct = {}
        if message is not None:
            dct['message'] = message
        if typeName is not None:
            dct['typeName'] = typeName
        if fullTypeName is not None:
            dct['fullTypeName'] = fullTypeName
        if evaluateName is not None:
            dct['evaluateName'] = evaluateName
        if stackTrace is not None:
            dct['stackTrace'] = stackTrace
        if innerException is not None:
            dct['innerException'] = [ExceptionDetails.update_dict_ids_to_dap(o) for o in innerException] if update_ids_to_dap and innerException else innerException
        dct.update(self.kwargs)
        return dct