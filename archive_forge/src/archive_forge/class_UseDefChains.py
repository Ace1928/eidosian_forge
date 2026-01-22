from pythran.passmanager import ModuleAnalysis
import pythran.metadata as md
import beniget
class UseDefChains(ModuleAnalysis):
    """
    Build use-define chains analysis for each variable.
    """

    def __init__(self):
        self.result = None
        super(UseDefChains, self).__init__(DefUseChains)

    def visit_Module(self, node):
        udc = beniget.UseDefChains(self.def_use_chains)
        self.result = udc.chains