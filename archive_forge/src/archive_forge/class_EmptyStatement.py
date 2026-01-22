from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class EmptyStatement(Statement):

    def __init__(self):
        Statement.__init__(self, '')