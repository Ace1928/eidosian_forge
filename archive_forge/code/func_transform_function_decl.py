from sympy.external import import_module
import os
def transform_function_decl(self, node):
    """Transformation Function For Function Declaration

            Used to create nodes for function declarations and definitions for
            the respective nodes in the clang AST

            Returns
            =======

            function : Codegen AST node
                - FunctionPrototype node if function body is not present
                - FunctionDefinition node if the function body is present


            """
    if node.result_type.kind in self._data_types['int']:
        ret_type = self._data_types['int'][node.result_type.kind]
    elif node.result_type.kind in self._data_types['float']:
        ret_type = self._data_types['float'][node.result_type.kind]
    elif node.result_type.kind in self._data_types['bool']:
        ret_type = self._data_types['bool'][node.result_type.kind]
    elif node.result_type.kind in self._data_types['void']:
        ret_type = self._data_types['void'][node.result_type.kind]
    else:
        raise NotImplementedError('Only void, bool, int and float are supported')
    body = []
    param = []
    try:
        children = node.get_children()
        child = next(children)
        while child.kind == cin.CursorKind.NAMESPACE_REF:
            child = next(children)
        while child.kind == cin.CursorKind.TYPE_REF:
            child = next(children)
        try:
            while True:
                decl = self.transform(child)
                if child.kind == cin.CursorKind.PARM_DECL:
                    param.append(decl)
                elif child.kind == cin.CursorKind.COMPOUND_STMT:
                    for val in decl:
                        body.append(val)
                else:
                    body.append(decl)
                child = next(children)
        except StopIteration:
            pass
    except StopIteration:
        pass
    if body == []:
        function = FunctionPrototype(return_type=ret_type, name=node.spelling, parameters=param)
    else:
        function = FunctionDefinition(return_type=ret_type, name=node.spelling, parameters=param, body=body)
    return function