from sympy.external import import_module
def visit_Function(self, node):
    """Visitor Function for function Definitions

            Visits each function definition present in the ASR and creates a
            function definition node in the Python AST with all the elements of the
            given function

            The functions declare all the variables required as SymPy symbols in
            the function before the function definition

            This function also the call_visior_function to parse the contents of
            the function body

            """
    fn_args = [Variable(arg_iter.name) for arg_iter in node.args]
    fn_body = []
    fn_name = node.name
    for i in node.body:
        fn_ast = call_visitor(i)
    try:
        fn_body_expr = fn_ast
    except UnboundLocalError:
        fn_body_expr = []
    for sym in node.symtab.symbols:
        decl = call_visitor(node.symtab.symbols[sym])
        for symbols in decl:
            fn_body.append(symbols)
    for elem in fn_body_expr:
        fn_body.append(elem)
    fn_body.append(Return(Variable(node.return_var.name)))
    if isinstance(node.return_var.type, asr.Integer):
        ret_type = IntBaseType(String('integer'))
    elif isinstance(node.return_var.type, asr.Real):
        ret_type = FloatBaseType(String('real'))
    else:
        raise NotImplementedError('Data type not supported')
    new_node = FunctionDefinition(return_type=ret_type, name=fn_name, parameters=fn_args, body=fn_body)
    self._py_ast.append(new_node)