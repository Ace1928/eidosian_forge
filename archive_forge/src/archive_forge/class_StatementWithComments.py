from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class StatementWithComments(object):

    def __init__(self, stmt, comment):
        self.stmt = stmt
        self.comment = comment

    def generate(self):
        yield '// {}'.format(self.comment)
        for s in self.stmt.generate():
            yield s