from time import time
from ..api import Any, DelegatesTo, HasTraits, Int, Range
class global_value(new_style_value):

    def init(self):
        global gvalue
        gvalue = -1

    def do_set(self):
        global gvalue
        for i in range(n):
            gvalue = i

    def do_get(self):
        global gvalue
        for i in range(n):
            gvalue