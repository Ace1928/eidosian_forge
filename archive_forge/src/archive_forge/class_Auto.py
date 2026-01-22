import codecs, re
from reportlab.lib.utils import isSeq, isBytes, isStr
from reportlab.lib import colors
from reportlab.lib.normalDate import NormalDate
class Auto(Validator):

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def test(self, x):
        return x is self.__class__ or isinstance(x, self.__class__)