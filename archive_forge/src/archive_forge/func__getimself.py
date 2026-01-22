import sys
def _getimself(function):
    return getattr(function, 'im_self', None)