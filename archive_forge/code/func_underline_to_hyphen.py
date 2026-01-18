from __future__ import absolute_import, division, print_function
import time
def underline_to_hyphen(input):
    if input and isinstance(input, str):
        return input.replace('_', '-')
    return input