from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class If(object):

    def __init__(self, condition, then_, else_=None):
        self.condition = condition
        self.then_ = then_
        self.else_ = else_

    def generate(self):
        yield ('if (%s)' % self.condition)
        for line in self.then_.generate():
            yield line
        if self.else_ is not None:
            yield 'else'
            for line in self.else_.generate():
                yield line