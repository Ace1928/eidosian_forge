from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
def add_capsule(self, func, ptrname, sig):
    self.capsules.append((ptrname, sig))
    self.implems.append(func)