import sys
def _totext(obj, encoding=None, errors=None):
    if isinstance(obj, bytes):
        if errors is None:
            obj = obj.decode(encoding)
        else:
            obj = obj.decode(encoding, errors)
    elif not isinstance(obj, str):
        obj = str(obj)
    return obj