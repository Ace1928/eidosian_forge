from __future__ import annotations
import numba as nb
import numpy as np
import os
def validate_operator(how, is_image):
    name = how if is_image else how + '_arr'
    if is_image:
        if name not in image_operators:
            raise ValueError('Operator %r not one of the supported image operators: %s' % (how, ', '.join((repr(el) for el in image_operators))))
    elif name not in array_operators:
        raise ValueError('Operator %r not one of the supported array operators: %s' % (how, ', '.join((repr(el[:-4]) for el in array_operators))))