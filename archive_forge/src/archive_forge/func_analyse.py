import gast as ast
from copy import deepcopy
from numpy import floating, integer, complexfloating
from pythran.tables import MODULES, attributes
import pythran.typing as typing
from pythran.syntax import PythranSyntaxError
from pythran.utils import isnum
def analyse(node, env, non_generic=None):
    """Computes the type of the expression given by node.

    The type of the node is computed in the context of the context of the
    supplied type environment env. Data types can be introduced into the
    language simply by having a predefined set of identifiers in the initial
    environment. Environment; this way there is no need to change the syntax
    or more importantly, the type-checking program when extending the language.

    Args:
        node: The root of the abstract syntax tree.
        env: The type environment is a mapping of expression identifier names
            to type assignments.
        non_generic: A set of non-generic variables, or None

    Returns:
        The computed type of the expression.

    Raises:
        InferenceError: The type of the expression could not be inferred,
        PythranTypeError: InferenceError with user friendly message + location
    """
    if non_generic is None:
        non_generic = set()
    if isinstance(node, ast.Name):
        if isinstance(node.ctx, ast.Store):
            new_type = TypeVariable()
            non_generic.add(new_type)
            env[node.id] = new_type
        return get_type(node.id, env, non_generic)
    elif isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            return Str()
        elif isinstance(node.value, int):
            return Integer()
        elif isinstance(node.value, float):
            return Float()
        elif isinstance(node.value, complex):
            return Complex()
        elif node.value is None:
            return NoneType
        else:
            raise NotImplementedError
    elif isinstance(node, ast.Compare):
        left_type = analyse(node.left, env, non_generic)
        comparators_type = [analyse(comparator, env, non_generic) for comparator in node.comparators]
        ops_type = [analyse(op, env, non_generic) for op in node.ops]
        prev_type = left_type
        result_type = TypeVariable()
        for op_type, comparator_type in zip(ops_type, comparators_type):
            try:
                unify(Function([prev_type, comparator_type], result_type), op_type)
                prev_type = comparator_type
            except InferenceError:
                raise PythranTypeError('Invalid comparison, between `{}` and `{}`'.format(prev_type, comparator_type), node)
        return result_type
    elif isinstance(node, ast.Call):
        if is_getattr(node):
            self_type = analyse(node.args[0], env, non_generic)
            attr_name = node.args[1].value
            _, attr_signature = attributes[attr_name]
            attr_type = tr(attr_signature)
            result_type = TypeVariable()
            try:
                unify(Function([self_type], result_type), attr_type)
            except InferenceError:
                if isinstance(prune(attr_type), MultiType):
                    msg = 'no attribute found, tried:\n{}'.format(attr_type)
                else:
                    msg = 'tried {}'.format(attr_type)
                raise PythranTypeError('Invalid attribute for getattr call with selfof type `{}`, {}'.format(self_type, msg), node)
        else:
            fun_type = analyse(node.func, env, non_generic)
            arg_types = [analyse(arg, env, non_generic) for arg in node.args]
            result_type = TypeVariable()
            try:
                unify(Function(arg_types, result_type), fun_type)
            except InferenceError:
                fun_type = analyse(node.func, env, non_generic)
                if isinstance(prune(fun_type), MultiType):
                    msg = 'no overload found, tried:\n{}'.format(fun_type)
                else:
                    msg = 'tried {}'.format(fun_type)
                raise PythranTypeError('Invalid argument type for function call to `Callable[[{}], ...]`, {}'.format(', '.join(('{}'.format(at) for at in arg_types)), msg), node)
        return result_type
    elif isinstance(node, ast.IfExp):
        test_type = analyse(node.test, env, non_generic)
        unify(Function([test_type], Bool()), tr(MODULES['builtins']['bool']))
        if is_test_is_none(node.test):
            none_id = node.test.left.id
            body_env = env.copy()
            body_env[none_id] = NoneType
        else:
            none_id = None
            body_env = env
        body_type = analyse(node.body, body_env, non_generic)
        if none_id:
            orelse_env = env.copy()
            if is_option_type(env[none_id]):
                orelse_env[none_id] = prune(env[none_id]).types[0]
            else:
                orelse_env[none_id] = TypeVariable()
        else:
            orelse_env = env
        orelse_type = analyse(node.orelse, orelse_env, non_generic)
        try:
            return merge_unify(body_type, orelse_type)
        except InferenceError:
            raise PythranTypeError('Incompatible types from different branches:`{}` and `{}`'.format(body_type, orelse_type), node)
    elif isinstance(node, ast.UnaryOp):
        operand_type = analyse(node.operand, env, non_generic)
        op_type = analyse(node.op, env, non_generic)
        result_type = TypeVariable()
        try:
            unify(Function([operand_type], result_type), op_type)
            return result_type
        except InferenceError:
            raise PythranTypeError('Invalid operand for `{}`: `{}`'.format(symbol_of[type(node.op)], operand_type), node)
    elif isinstance(node, ast.BinOp):
        left_type = analyse(node.left, env, non_generic)
        op_type = analyse(node.op, env, non_generic)
        right_type = analyse(node.right, env, non_generic)
        result_type = TypeVariable()
        try:
            unify(Function([left_type, right_type], result_type), op_type)
        except InferenceError:
            raise PythranTypeError('Invalid operand for `{}`: `{}` and `{}`'.format(symbol_of[type(node.op)], left_type, right_type), node)
        return result_type
    elif isinstance(node, ast.Pow):
        return tr(MODULES['numpy']['power'])
    elif isinstance(node, ast.Sub):
        return tr(MODULES['operator']['sub'])
    elif isinstance(node, (ast.USub, ast.UAdd)):
        return tr(MODULES['operator']['pos'])
    elif isinstance(node, (ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot)):
        return tr(MODULES['operator']['eq'])
    elif isinstance(node, (ast.In, ast.NotIn)):
        contains_sig = tr(MODULES['operator']['contains'])
        contains_sig.types[:-1] = reversed(contains_sig.types[:-1])
        return contains_sig
    elif isinstance(node, ast.Add):
        return tr(MODULES['operator']['add'])
    elif isinstance(node, ast.Mult):
        return tr(MODULES['operator']['mul'])
    elif isinstance(node, ast.MatMult):
        return tr(MODULES['operator']['matmul'])
    elif isinstance(node, (ast.Div, ast.FloorDiv)):
        return tr(MODULES['operator']['floordiv'])
    elif isinstance(node, ast.Mod):
        return tr(MODULES['operator']['mod'])
    elif isinstance(node, (ast.LShift, ast.RShift)):
        return tr(MODULES['operator']['lshift'])
    elif isinstance(node, (ast.BitXor, ast.BitAnd, ast.BitOr)):
        return tr(MODULES['operator']['lshift'])
    elif isinstance(node, ast.List):
        new_type = TypeVariable()
        for elt in node.elts:
            elt_type = analyse(elt, env, non_generic)
            try:
                unify(new_type, elt_type)
            except InferenceError:
                raise PythranTypeError('Incompatible list element type `{}` and `{}`'.format(new_type, elt_type), node)
        return List(new_type)
    elif isinstance(node, ast.Set):
        new_type = TypeVariable()
        for elt in node.elts:
            elt_type = analyse(elt, env, non_generic)
            try:
                unify(new_type, elt_type)
            except InferenceError:
                raise PythranTypeError('Incompatible set element type `{}` and `{}`'.format(new_type, elt_type), node)
        return Set(new_type)
    elif isinstance(node, ast.Dict):
        new_key_type = TypeVariable()
        for key in node.keys:
            key_type = analyse(key, env, non_generic)
            try:
                unify(new_key_type, key_type)
            except InferenceError:
                raise PythranTypeError('Incompatible dict key type `{}` and `{}`'.format(new_key_type, key_type), node)
        new_value_type = TypeVariable()
        for value in node.values:
            value_type = analyse(value, env, non_generic)
            try:
                unify(new_value_type, value_type)
            except InferenceError:
                raise PythranTypeError('Incompatible dict value type `{}` and `{}`'.format(new_value_type, value_type), node)
        return Dict(new_key_type, new_value_type)
    elif isinstance(node, ast.Tuple):
        return Tuple([analyse(elt, env, non_generic) for elt in node.elts])
    elif isinstance(node, ast.Slice):

        def unify_int_or_none(t, name):
            try:
                unify(t, Integer())
            except InferenceError:
                try:
                    unify(t, NoneType)
                except InferenceError:
                    raise PythranTypeError('Invalid slice {} type `{}`, expecting int or None'.format(name, t))
        if node.lower:
            lower_type = analyse(node.lower, env, non_generic)
            unify_int_or_none(lower_type, 'lower bound')
        else:
            lower_type = Integer()
        if node.upper:
            upper_type = analyse(node.upper, env, non_generic)
            unify_int_or_none(upper_type, 'upper bound')
        else:
            upper_type = Integer()
        if node.step:
            step_type = analyse(node.step, env, non_generic)
            unify_int_or_none(step_type, 'step')
        else:
            step_type = Integer()
        return Slice
    elif isinstance(node, ast.Subscript):
        new_type = TypeVariable()
        value_type = prune(analyse(node.value, env, non_generic))
        try:
            slice_type = prune(analyse(node.slice, env, non_generic))
        except PythranTypeError as e:
            raise PythranTypeError(e.msg, node)
        if isinstance(node.slice, ast.Tuple):
            nbslice = len(node.slice.elts)
            dtype = TypeVariable()
            try:
                unify(Array(dtype, nbslice), clone(value_type))
            except InferenceError:
                raise PythranTypeError('Dimension mismatch when slicing `{}`'.format(value_type), node)
            return TypeVariable()
        else:
            num = isnum(node.slice)
            if num and is_tuple_type(value_type):
                try:
                    unify(prune(prune(value_type.types[0]).types[0]).types[node.slice.value], new_type)
                    return new_type
                except IndexError:
                    raise PythranTypeError('Invalid tuple indexing, out-of-bound index `{}` for type `{}`'.format(node.slice.value, value_type), node)
        try:
            unify(tr(MODULES['operator']['getitem']), Function([value_type, slice_type], new_type))
        except InferenceError:
            raise PythranTypeError('Invalid subscripting of `{}` by `{}`'.format(value_type, slice_type), node)
        return new_type
        return new_type
    elif isinstance(node, ast.Attribute):
        from pythran.utils import attr_to_path
        obj, path = attr_to_path(node)
        if obj.signature is typing.Any:
            return TypeVariable()
        else:
            return tr(obj)
    elif isinstance(node, ast.Import):
        for alias in node.names:
            if alias.name not in MODULES:
                raise NotImplementedError('unknown module: %s ' % alias.name)
            if alias.asname is None:
                target = alias.name
            else:
                target = alias.asname
            env[target] = tr(MODULES[alias.name])
        return env
    elif isinstance(node, ast.ImportFrom):
        if node.module not in MODULES:
            raise NotImplementedError('unknown module: %s' % node.module)
        for alias in node.names:
            if alias.name not in MODULES[node.module]:
                raise NotImplementedError('unknown function: %s in %s' % (alias.name, node.module))
            if alias.asname is None:
                target = alias.name
            else:
                target = alias.asname
            env[target] = tr(MODULES[node.module][alias.name])
        return env
    elif isinstance(node, ast.FunctionDef):
        ftypes = []
        for i in range(1 + len(node.args.defaults)):
            new_env = env.copy()
            new_non_generic = non_generic.copy()
            new_env.pop('@ret', None)
            new_env.pop('@gen', None)
            hy = HasYield()
            for stmt in node.body:
                hy.visit(stmt)
            new_env['@gen'] = hy.has_yield
            arg_types = []
            istop = len(node.args.args) - i
            for arg in node.args.args[:istop]:
                arg_type = TypeVariable()
                new_env[arg.id] = arg_type
                new_non_generic.add(arg_type)
                arg_types.append(arg_type)
            for arg, expr in zip(node.args.args[istop:], node.args.defaults[-i:]):
                arg_type = analyse(expr, new_env, new_non_generic)
                new_env[arg.id] = arg_type
            analyse_body(node.body, new_env, new_non_generic)
            result_type = new_env.get('@ret', NoneType)
            if new_env['@gen']:
                result_type = Generator(result_type)
            ftype = Function(arg_types, result_type)
            ftypes.append(ftype)
        if len(ftypes) == 1:
            ftype = ftypes[0]
            env[node.name] = ftype
        else:
            env[node.name] = MultiType(ftypes)
        return env
    elif isinstance(node, ast.Module):
        analyse_body(node.body, env, non_generic)
        return env
    elif isinstance(node, (ast.Pass, ast.Break, ast.Continue)):
        return env
    elif isinstance(node, ast.Expr):
        analyse(node.value, env, non_generic)
        return env
    elif isinstance(node, ast.Delete):
        for target in node.targets:
            if isinstance(target, ast.Name):
                if target.id in env:
                    del env[target.id]
                else:
                    raise PythranTypeError('Invalid del: unbound identifier `{}`'.format(target.id), node)
            else:
                analyse(target, env, non_generic)
        return env
    elif isinstance(node, ast.Print):
        if node.dest is not None:
            analyse(node.dest, env, non_generic)
        for value in node.values:
            analyse(value, env, non_generic)
        return env
    elif isinstance(node, ast.Assign):
        defn_type = analyse(node.value, env, non_generic)
        for target in node.targets:
            target_type = analyse(target, env, non_generic)
            try:
                unify(target_type, defn_type)
            except InferenceError:
                raise PythranTypeError('Invalid assignment from type `{}` to type `{}`'.format(target_type, defn_type), node)
        return env
    elif isinstance(node, ast.AugAssign):
        fake_target = deepcopy(node.target)
        fake_target.ctx = ast.Load()
        fake_op = ast.BinOp(fake_target, node.op, node.value)
        ast.copy_location(fake_op, node)
        res_type = analyse(fake_op, env, non_generic)
        target_type = analyse(node.target, env, non_generic)
        try:
            unify(target_type, res_type)
        except InferenceError:
            raise PythranTypeError('Invalid update operand for `{}`: `{}` and `{}`'.format(symbol_of[type(node.op)], res_type, target_type), node)
        return env
    elif isinstance(node, ast.Raise):
        return env
    elif isinstance(node, ast.Return):
        if env['@gen']:
            return env
        if node.value is None:
            ret_type = NoneType
        else:
            ret_type = analyse(node.value, env, non_generic)
        if '@ret' in env:
            try:
                ret_type = merge_unify(env['@ret'], ret_type)
            except InferenceError:
                raise PythranTypeError('function may returns with incompatible types `{}` and `{}`'.format(env['@ret'], ret_type), node)
        env['@ret'] = ret_type
        return env
    elif isinstance(node, ast.Yield):
        assert env['@gen']
        assert node.value is not None
        if node.value is None:
            ret_type = NoneType
        else:
            ret_type = analyse(node.value, env, non_generic)
        if '@ret' in env:
            try:
                ret_type = merge_unify(env['@ret'], ret_type)
            except InferenceError:
                raise PythranTypeError('function may yields incompatible types `{}` and `{}`'.format(env['@ret'], ret_type), node)
        env['@ret'] = ret_type
        return env
    elif isinstance(node, ast.For):
        iter_type = analyse(node.iter, env, non_generic)
        target_type = analyse(node.target, env, non_generic)
        unify(Collection(TypeVariable(), TypeVariable(), TypeVariable(), target_type), iter_type)
        analyse_body(node.body, env, non_generic)
        analyse_body(node.orelse, env, non_generic)
        return env
    elif isinstance(node, ast.If):
        test_type = analyse(node.test, env, non_generic)
        unify(Function([test_type], Bool()), tr(MODULES['builtins']['bool']))
        body_env = env.copy()
        body_non_generic = non_generic.copy()
        if is_test_is_none(node.test):
            none_id = node.test.left.id
            body_env[none_id] = NoneType
        else:
            none_id = None
        analyse_body(node.body, body_env, body_non_generic)
        orelse_env = env.copy()
        orelse_non_generic = non_generic.copy()
        if none_id:
            if is_option_type(env[none_id]):
                orelse_env[none_id] = prune(env[none_id]).types[0]
            else:
                orelse_env[none_id] = TypeVariable()
        analyse_body(node.orelse, orelse_env, orelse_non_generic)
        for var in body_env:
            if var not in env:
                if var in orelse_env:
                    try:
                        new_type = merge_unify(body_env[var], orelse_env[var])
                    except InferenceError:
                        raise PythranTypeError('Incompatible types from different branches for `{}`: `{}` and `{}`'.format(var, body_env[var], orelse_env[var]), node)
                else:
                    new_type = body_env[var]
                env[var] = new_type
        for var in orelse_env:
            if var not in env:
                if var in body_env:
                    new_type = merge_unify(orelse_env[var], body_env[var])
                else:
                    new_type = orelse_env[var]
                env[var] = new_type
        if none_id:
            try:
                new_type = merge_unify(body_env[none_id], orelse_env[none_id])
            except InferenceError:
                msg = 'Inconsistent types while merging values of `{}` from conditional branches: `{}` and `{}`'
                err = msg.format(none_id, body_env[none_id], orelse_env[none_id])
                raise PythranTypeError(err, node)
            env[none_id] = new_type
        return env
    elif isinstance(node, ast.While):
        test_type = analyse(node.test, env, non_generic)
        unify(Function([test_type], Bool()), tr(MODULES['builtins']['bool']))
        analyse_body(node.body, env, non_generic)
        analyse_body(node.orelse, env, non_generic)
        return env
    elif isinstance(node, ast.Try):
        analyse_body(node.body, env, non_generic)
        for handler in node.handlers:
            analyse(handler, env, non_generic)
        analyse_body(node.orelse, env, non_generic)
        analyse_body(node.finalbody, env, non_generic)
        return env
    elif isinstance(node, ast.ExceptHandler):
        if node.name:
            new_type = ExceptionType
            non_generic.add(new_type)
            if node.name.id in env:
                unify(env[node.name.id], new_type)
            else:
                env[node.name.id] = new_type
        analyse_body(node.body, env, non_generic)
        return env
    elif isinstance(node, ast.Assert):
        if node.msg:
            analyse(node.msg, env, non_generic)
        analyse(node.test, env, non_generic)
        return env
    elif isinstance(node, ast.UnaryOp):
        operand_type = analyse(node.operand, env, non_generic)
        return_type = TypeVariable()
        op_type = analyse(node.op, env, non_generic)
        unify(Function([operand_type], return_type), op_type)
        return return_type
    elif isinstance(node, ast.Invert):
        return MultiType([Function([Bool()], Integer()), Function([Integer()], Integer())])
    elif isinstance(node, ast.Not):
        return tr(MODULES['builtins']['bool'])
    elif isinstance(node, ast.BoolOp):
        op_type = analyse(node.op, env, non_generic)
        value_types = [analyse(value, env, non_generic) for value in node.values]
        for value_type in value_types:
            unify(Function([value_type], Bool()), tr(MODULES['builtins']['bool']))
        return_type = TypeVariable()
        prev_type = value_types[0]
        for value_type in value_types[1:]:
            unify(Function([prev_type, value_type], return_type), op_type)
            prev_type = value_type
        return return_type
    elif isinstance(node, (ast.And, ast.Or)):
        x_type = TypeVariable()
        return MultiType([Function([x_type, x_type], x_type), Function([TypeVariable(), TypeVariable()], TypeVariable())])
    raise RuntimeError('Unhandled syntax node {0}'.format(type(node)))