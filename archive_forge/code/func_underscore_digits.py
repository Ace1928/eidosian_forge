from __future__ import absolute_import, unicode_literals
def underscore_digits(d):
    return Rep1(d) + Rep(Str('_') + Rep1(d))