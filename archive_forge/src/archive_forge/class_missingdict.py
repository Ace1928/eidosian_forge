from fontTools.ttLib import TTFont
from collections import defaultdict
from operator import add
from functools import reduce
class missingdict(dict):

    def __init__(self, missing_func):
        self.missing_func = missing_func

    def __missing__(self, v):
        return self.missing_func(v)