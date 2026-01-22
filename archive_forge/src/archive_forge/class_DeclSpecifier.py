from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class DeclSpecifier(NestedDeclarator):

    def __init__(self, subdecl, spec, sep=' '):
        NestedDeclarator.__init__(self, subdecl)
        self.spec = spec
        self.sep = sep

    def get_decl_pair(self):

        def add_spec(sub_it):
            it = iter(sub_it)
            try:
                yield ('%s%s%s' % (self.spec, self.sep, next(it)))
            except StopIteration:
                pass
            for line in it:
                yield line
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        return (add_spec(sub_tp), sub_decl)