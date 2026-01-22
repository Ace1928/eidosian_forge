from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class Define(object):

    def __init__(self, symbol, value):
        self.symbol = symbol
        self.value = value

    def generate(self):
        yield ('#define %s %s' % (self.symbol, self.value))