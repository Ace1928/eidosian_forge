from pythran.passmanager import ModuleAnalysis
def visit_metadata(self, node):
    if hasattr(node, 'metadata'):
        self.generic_visit(node.metadata)