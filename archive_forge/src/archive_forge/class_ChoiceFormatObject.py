import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
class ChoiceFormatObject(NAryFormatObject):

    def is_choice(sef):
        return True

    def as_tuple(self):
        return ('choice', [a.as_tuple() for a in self.children])

    def space_upto_nl(self):
        return self.children[0].space_upto_nl()

    def flat(self):
        return self.children[0].flat()