import os
import re
import random
from gimpfu import *
def v0_pattern_pixel(char, alpha, fmt):
    if fmt == 'rgba':
        if char == 't':
            return [0, 0, 0, 0]
        return v0_PIXELS[char] + [alpha]
    if fmt == 'rgb':
        if char == 't':
            return [0, 0, 0]
        return v0_PIXELS[char]
    if fmt == 'gray':
        assert char in '0123456789ABCDEF'
        return [v0_PIXELS[char][0]]
    if fmt == 'graya':
        assert char in 't0123456789ABCDEF'
        if char == 't':
            return [0, 0]
        return [v0_PIXELS[char][0]] + [alpha]
    raise Exception('v0_pattern_pixel: unknown format {}'.format(fmt))