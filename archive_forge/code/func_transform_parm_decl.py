from sympy.external import import_module
import os
def transform_parm_decl(self, node):
    """Transformation function for Parameter Declaration

            Used to create parameter nodes for the required functions for the
            respective nodes in the clang AST

            Returns
            =======

            param : Codegen AST Node
                Variable node with the value and type of the variable

            Raises
            ======

            ValueError if multiple children encountered in the parameter node

            """
    if node.type.kind in self._data_types['int']:
        type = self._data_types['int'][node.type.kind]
    elif node.type.kind in self._data_types['float']:
        type = self._data_types['float'][node.type.kind]
    elif node.type.kind in self._data_types['bool']:
        type = self._data_types['bool'][node.type.kind]
    else:
        raise NotImplementedError('Only bool, int and float are supported')
    try:
        children = node.get_children()
        child = next(children)
        while child.kind in [cin.CursorKind.NAMESPACE_REF, cin.CursorKind.TYPE_REF, cin.CursorKind.TEMPLATE_REF]:
            child = next(children)
        lit = self.transform(child)
        if node.type.kind in self._data_types['int']:
            val = Integer(lit)
        elif node.type.kind in self._data_types['float']:
            val = Float(lit)
        elif node.type.kind in self._data_types['bool']:
            val = sympify(bool(lit))
        else:
            raise NotImplementedError('Only bool, int and float are supported')
        param = Variable(node.spelling).as_Declaration(type=type, value=val)
    except StopIteration:
        param = Variable(node.spelling).as_Declaration(type=type)
    try:
        self.transform(next(children))
        raise ValueError("Can't handle multiple children on parameter")
    except StopIteration:
        pass
    return param