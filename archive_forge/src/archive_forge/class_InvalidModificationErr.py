from .domreg import getDOMImplementation, registerDOMImplementation
class InvalidModificationErr(DOMException):
    code = INVALID_MODIFICATION_ERR