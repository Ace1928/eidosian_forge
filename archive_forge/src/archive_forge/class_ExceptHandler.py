from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class ExceptHandler(object):

    def __init__(self, name, body, alias=None):
        self.name = name
        self.body = body
        self.alias = alias

    def generate(self):
        if self.name is None:
            yield 'catch(...)'
        else:
            yield ('catch (pythonic::types::%s const& %s)' % (self.name, self.alias or ''))
        for line in self.body.generate():
            yield line