from pythran.passmanager import ModuleAnalysis
import pythran.metadata as md
import beniget
class ExtendedDefUseChains(beniget.DefUseChains):

    def unbound_identifier(self, name, node):
        pass

    def visit(self, node):
        md.visit(self, node)
        return super(ExtendedDefUseChains, self).visit(node)