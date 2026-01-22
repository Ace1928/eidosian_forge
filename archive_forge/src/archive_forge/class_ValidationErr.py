from .domreg import getDOMImplementation, registerDOMImplementation
class ValidationErr(DOMException):
    code = VALIDATION_ERR