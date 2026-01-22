from parso.python import tree
from jedi import debug
from jedi.inference.helpers import is_string
def _check_for_exception_catch(node_context, jedi_name, exception, payload=None):
    """
    Checks if a jedi object (e.g. `Statement`) sits inside a try/catch and
    doesn't count as an error (if equal to `exception`).
    Also checks `hasattr` for AttributeErrors and uses the `payload` to compare
    it.
    Returns True if the exception was catched.
    """

    def check_match(cls, exception):
        if not cls.is_class():
            return False
        for python_cls in exception.mro():
            if cls.py__name__() == python_cls.__name__ and cls.parent_context.is_builtins_module():
                return True
        return False

    def check_try_for_except(obj, exception):
        iterator = iter(obj.children)
        for branch_type in iterator:
            next(iterator)
            suite = next(iterator)
            if branch_type == 'try' and (not branch_type.start_pos < jedi_name.start_pos <= suite.end_pos):
                return False
        for node in obj.get_except_clause_tests():
            if node is None:
                return True
            else:
                except_classes = node_context.infer_node(node)
                for cls in except_classes:
                    from jedi.inference.value import iterable
                    if isinstance(cls, iterable.Sequence) and cls.array_type == 'tuple':
                        for lazy_value in cls.py__iter__():
                            for typ in lazy_value.infer():
                                if check_match(typ, exception):
                                    return True
                    elif check_match(cls, exception):
                        return True

    def check_hasattr(node, suite):
        try:
            assert suite.start_pos <= jedi_name.start_pos < suite.end_pos
            assert node.type in ('power', 'atom_expr')
            base = node.children[0]
            assert base.type == 'name' and base.value == 'hasattr'
            trailer = node.children[1]
            assert trailer.type == 'trailer'
            arglist = trailer.children[1]
            assert arglist.type == 'arglist'
            from jedi.inference.arguments import TreeArguments
            args = TreeArguments(node_context.inference_state, node_context, arglist)
            unpacked_args = list(args.unpack())
            assert len(unpacked_args) == 2
            key, lazy_value = unpacked_args[1]
            names = list(lazy_value.infer())
            assert len(names) == 1 and is_string(names[0])
            assert names[0].get_safe_value() == payload[1].value
            key, lazy_value = unpacked_args[0]
            objects = lazy_value.infer()
            return payload[0] in objects
        except AssertionError:
            return False
    obj = jedi_name
    while obj is not None and (not isinstance(obj, (tree.Function, tree.Class))):
        if isinstance(obj, tree.Flow):
            if obj.type == 'try_stmt' and check_try_for_except(obj, exception):
                return True
            if exception == AttributeError and obj.type in ('if_stmt', 'while_stmt'):
                if check_hasattr(obj.children[1], obj.children[3]):
                    return True
        obj = obj.parent
    return False