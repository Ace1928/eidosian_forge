from __future__ import absolute_import, division, print_function
import sys
def _convert_int_to_bytes(count, n):
    h = '%x' % n
    if len(h) > 2 * count:
        raise Exception('Number {1} needs more than {0} bytes!'.format(count, n))
    return ('0' * (2 * count - len(h)) + h).decode('hex')