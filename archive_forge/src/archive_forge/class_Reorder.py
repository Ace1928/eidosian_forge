import gast as ast
from pythran.analyses import OrderedGlobalDeclarations
from pythran.passmanager import Transformation
from pythran.syntax import PythranSyntaxError
from pythran.types.type_dependencies import TypeDependencies
import pythran.graph as graph
class Reorder(Transformation):
    """ Reorder top-level functions to prevent circular type dependencies.  """

    def __init__(self):
        """ Trigger others analysis informations. """
        super(Reorder, self).__init__(TypeDependencies, OrderedGlobalDeclarations)

    def prepare(self, node):
        """ Format type dependencies information to use if for reordering. """
        super(Reorder, self).prepare(node)
        candidates = self.type_dependencies.successors(TypeDependencies.NoDeps)
        while candidates:
            new_candidates = list()
            for n in candidates:
                for p in list(self.type_dependencies.predecessors(n)):
                    if graph.has_path(self.type_dependencies, n, p):
                        self.type_dependencies.remove_edge(p, n)
                if n not in self.type_dependencies.successors(n):
                    new_candidates.extend(self.type_dependencies.successors(n))
            candidates = new_candidates

    def visit_Module(self, node):
        """
        Keep everything but function definition then add sorted functions.

        Most of the time, many function sort work so we use function calldepth
        as a "sort hint" to simplify typing.
        """
        newbody = list()
        olddef = list()
        for stmt in node.body:
            if isinstance(stmt, ast.FunctionDef):
                olddef.append(stmt)
            else:
                newbody.append(stmt)
        try:
            newdef = topological_sort(self.type_dependencies, self.ordered_global_declarations)
            newdef = [f for f in newdef if isinstance(f, ast.FunctionDef)]
        except graph.Unfeasible:
            raise PythranSyntaxError('Infinite function recursion')
        assert set(newdef) == set(olddef), 'A function have been lost...'
        node.body = newbody + newdef
        self.update = True
        return node