from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
def add_pyfunction(self, func, name, types, signature):
    self.add_function_to(self.python_implems, func, name, types, signature)