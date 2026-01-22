import sys
import json
from .symbols import *
from .symbols import Symbol
class JsonLoader(object):

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, src):
        if isinstance(src, string_types):
            return json.loads(src, **self.kwargs)
        else:
            return json.load(src, **self.kwargs)