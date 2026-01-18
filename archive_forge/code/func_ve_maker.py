import io
import math
import os
import typing
import weakref
def ve_maker(ve):
    if type(ve) not in (list, tuple) or len(ve) < 2:
        raise ValueError("bad 've' format: %s" % ve)
    if ve[0].lower() not in ('and', 'or', 'not'):
        raise ValueError('bad operand: %s' % ve[0])
    if ve[0].lower() == 'not' and len(ve) != 2:
        raise ValueError("bad 've' format: %s" % ve)
    item = '[/%s' % ve[0].title()
    for x in ve[1:]:
        if type(x) is int:
            if x not in all_ocgs:
                raise ValueError('bad OCG %i' % x)
            item += ' %i 0 R' % x
        else:
            item += ' %s' % ve_maker(x)
    item += ']'
    return item