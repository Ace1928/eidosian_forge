from . import c_ast
def _fix_atomic_specifiers_once(decl):
    """ Performs one 'fix' round of atomic specifiers.
        Returns (modified_decl, found) where found is True iff a fix was made.
    """
    parent = decl
    grandparent = None
    node = decl.type
    while node is not None:
        if isinstance(node, c_ast.Typename) and '_Atomic' in node.quals:
            break
        try:
            grandparent = parent
            parent = node
            node = node.type
        except AttributeError:
            return (decl, False)
    assert isinstance(parent, c_ast.TypeDecl)
    grandparent.type = node.type
    if '_Atomic' not in node.type.quals:
        node.type.quals.append('_Atomic')
    return (decl, True)