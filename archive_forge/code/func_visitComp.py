from pythran.analyses import OptimizableComprehension
from pythran.passmanager import Transformation
from pythran.transformations.normalize_tuples import ConvertToTuple
from pythran.conversion import mangle
from pythran.utils import attr_to_path, path_to_attr
import gast as ast
def visitComp(self, node, make_attr):
    if node in self.optimizable_comprehension:
        self.update = True
        self.generic_visit(node)
        iters = [self.make_Iterator(gen) for gen in node.generators]
        variables = [ast.Name(gen.target.id, ast.Param(), None, None) for gen in node.generators]
        if len(iters) == 1:
            iterAST = iters[0]
            varAST = ast.arguments([variables[0]], [], None, [], [], None, [])
        else:
            self.use_itertools = True
            prodName = ast.Attribute(value=ast.Name(id=mangle('itertools'), ctx=ast.Load(), annotation=None, type_comment=None), attr='product', ctx=ast.Load())
            varid = variables[0].id
            renamings = {v.id: (i,) for i, v in enumerate(variables)}
            node.elt = ConvertToTuple(varid, renamings).visit(node.elt)
            iterAST = ast.Call(prodName, iters, [])
            varAST = ast.arguments([ast.Name(varid, ast.Param(), None, None)], [], None, [], [], None, [])
        ldBodymap = node.elt
        ldmap = ast.Lambda(varAST, ldBodymap)
        return make_attr(ldmap, iterAST)
    else:
        return self.generic_visit(node)