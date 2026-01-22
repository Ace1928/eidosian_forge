from ._base import *
import operator as op
class LazyAny(object):

    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, v)