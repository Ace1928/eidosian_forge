from .domreg import getDOMImplementation, registerDOMImplementation
class NotFoundErr(DOMException):
    code = NOT_FOUND_ERR