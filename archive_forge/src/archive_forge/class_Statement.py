from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class Statement(object):

    def __init__(self, text):
        self.text = text

    def generate(self):
        yield (self.text + ';')