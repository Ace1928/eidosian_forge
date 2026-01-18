from sympy.external import import_module
import os
def transform_var_decl(self, node):
    """Transformation Function for Variable Declaration

            Used to create nodes for variable declarations and assignments with
            values or function call for the respective nodes in the clang AST

            Returns
            =======

            A variable node as Declaration, with the initial value if given

            Raises
            ======

            NotImplementedError : if called for data types not currently
            implemented

            Notes
            =====

            The function currently supports following data types:

            Boolean:
                bool, _Bool

            Integer:
                8-bit: signed char and unsigned char
                16-bit: short, short int, signed short,
                    signed short int, unsigned short, unsigned short int
                32-bit: int, signed int, unsigned int
                64-bit: long, long int, signed long,
                    signed long int, unsigned long, unsigned long int

            Floating point:
                Single Precision: float
                Double Precision: double
                Extended Precision: long double

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
        while child.kind == cin.CursorKind.NAMESPACE_REF:
            child = next(children)
        while child.kind == cin.CursorKind.TYPE_REF:
            child = next(children)
        val = self.transform(child)
        supported_rhs = [cin.CursorKind.INTEGER_LITERAL, cin.CursorKind.FLOATING_LITERAL, cin.CursorKind.UNEXPOSED_EXPR, cin.CursorKind.BINARY_OPERATOR, cin.CursorKind.PAREN_EXPR, cin.CursorKind.UNARY_OPERATOR, cin.CursorKind.CXX_BOOL_LITERAL_EXPR]
        if child.kind in supported_rhs:
            if isinstance(val, str):
                value = Symbol(val)
            elif isinstance(val, bool):
                if node.type.kind in self._data_types['int']:
                    value = Integer(0) if val == False else Integer(1)
                elif node.type.kind in self._data_types['float']:
                    value = Float(0.0) if val == False else Float(1.0)
                elif node.type.kind in self._data_types['bool']:
                    value = sympify(val)
            elif isinstance(val, (Integer, int, Float, float)):
                if node.type.kind in self._data_types['int']:
                    value = Integer(val)
                elif node.type.kind in self._data_types['float']:
                    value = Float(val)
                elif node.type.kind in self._data_types['bool']:
                    value = sympify(bool(val))
            else:
                value = val
            return Variable(node.spelling).as_Declaration(type=type, value=value)
        elif child.kind == cin.CursorKind.CALL_EXPR:
            return Variable(node.spelling).as_Declaration(value=val)
        else:
            raise NotImplementedError('Given variable declaration "{}" is not possible to parse yet!'.format(' '.join((t.spelling for t in node.get_tokens()))))
    except StopIteration:
        return Variable(node.spelling).as_Declaration(type=type)