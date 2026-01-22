from __future__ import absolute_import, division, print_function
class FMGBaseException(Exception):
    """Wrapper to catch the unexpected"""

    def __init__(self, msg=None, *args, **kwargs):
        if msg is None:
            msg = 'An exception occurred within the fortimanager.py httpapi connection plugin.'
        super(FMGBaseException, self).__init__(msg, *args)