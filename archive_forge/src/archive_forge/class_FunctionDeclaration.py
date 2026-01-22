from textwrap import dedent
from pythran.tables import pythran_ward
from pythran.spec import signatures_to_string
from pythran.utils import quote_cxxstring
class FunctionDeclaration(NestedDeclarator):

    def __init__(self, subdecl, arg_decls, *attributes):
        NestedDeclarator.__init__(self, subdecl)
        self.inline = True
        self.arg_decls = arg_decls
        self.attributes = attributes

    def get_decl_pair(self):
        sub_tp, sub_decl = self.subdecl.get_decl_pair()
        if self.inline:
            sub_tp = ['inline'] + sub_tp
        return (sub_tp, '%s(%s) %s' % (sub_decl, ', '.join((ad.inline() for ad in self.arg_decls)), ' '.join(self.attributes)))