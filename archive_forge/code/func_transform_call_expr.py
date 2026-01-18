from sympy.external import import_module
import os
def transform_call_expr(self, node):
    """Transformation function for a call expression

            Used to create function call nodes for the function calls present
            in the C code

            Returns
            =======

            FunctionCall : Codegen AST Node
                FunctionCall node with parameters if any parameters are present

            """
    param = []
    children = node.get_children()
    child = next(children)
    while child.kind == cin.CursorKind.NAMESPACE_REF:
        child = next(children)
    while child.kind == cin.CursorKind.TYPE_REF:
        child = next(children)
    first_child = self.transform(child)
    try:
        for child in children:
            arg = self.transform(child)
            if child.kind == cin.CursorKind.INTEGER_LITERAL:
                param.append(Integer(arg))
            elif child.kind == cin.CursorKind.FLOATING_LITERAL:
                param.append(Float(arg))
            else:
                param.append(arg)
        return FunctionCall(first_child, param)
    except StopIteration:
        return FunctionCall(first_child)