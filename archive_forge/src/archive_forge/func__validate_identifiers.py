import functools
import re
import warnings
@classmethod
def _validate_identifiers(cls, identifiers, allow_leading_zeroes=False):
    for item in identifiers:
        if not item:
            raise ValueError('Invalid empty identifier %r in %r' % (item, '.'.join(identifiers)))
        if item[0] == '0' and item.isdigit() and (item != '0') and (not allow_leading_zeroes):
            raise ValueError('Invalid leading zero in identifier %r' % item)