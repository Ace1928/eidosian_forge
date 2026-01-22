import ast
from collections import defaultdict, OrderedDict
import contextlib
import sys
from types import SimpleNamespace
import numpy as np
import operator
from numba.core import types, targetconfig, ir, rewrites, compiler
from numba.core.typing import npydecl
from numba.np.ufunc.dufunc import DUFunc
@rewrites.register_rewrite('after-inference')
class RewriteArrayExprs(rewrites.Rewrite):
    """The RewriteArrayExprs class is responsible for finding array
    expressions in Numba intermediate representation code, and
    rewriting those expressions to a single operation that will expand
    into something similar to a ufunc call.
    """

    def __init__(self, state, *args, **kws):
        super(RewriteArrayExprs, self).__init__(state, *args, **kws)
        special_ops = state.targetctx.special_ops
        if 'arrayexpr' not in special_ops:
            special_ops['arrayexpr'] = _lower_array_expr

    def match(self, func_ir, block, typemap, calltypes):
        """
        Using typing and a basic block, search the basic block for array
        expressions.
        Return True when one or more matches were found, False otherwise.
        """
        if len(calltypes) == 0:
            return False
        self.crnt_block = block
        self.typemap = typemap
        self.array_assigns = OrderedDict()
        self.const_assigns = {}
        assignments = block.find_insts(ir.Assign)
        for instr in assignments:
            target_name = instr.target.name
            expr = instr.value
            if isinstance(expr, ir.Expr) and isinstance(typemap.get(target_name, None), types.Array):
                self._match_array_expr(instr, expr, target_name)
            elif isinstance(expr, ir.Const):
                self.const_assigns[target_name] = expr
        return len(self.array_assigns) > 0

    def _match_array_expr(self, instr, expr, target_name):
        """
        Find whether the given assignment (*instr*) of an expression (*expr*)
        to variable *target_name* is an array expression.
        """
        expr_op = expr.op
        array_assigns = self.array_assigns
        if expr_op in ('unary', 'binop') and expr.fn in npydecl.supported_array_operators:
            if all((self.typemap[var.name].is_internal for var in expr.list_vars())):
                array_assigns[target_name] = instr
        elif expr_op == 'call' and expr.func.name in self.typemap:
            func_type = self.typemap[expr.func.name]
            if isinstance(func_type, types.Function):
                func_key = func_type.typing_key
                if _is_ufunc(func_key):
                    if not self._has_explicit_output(expr, func_key):
                        array_assigns[target_name] = instr

    def _has_explicit_output(self, expr, func):
        """
        Return whether the *expr* call to *func* (a ufunc) features an
        explicit output argument.
        """
        nargs = len(expr.args) + len(expr.kws)
        if expr.vararg is not None:
            return True
        return nargs > func.nin

    def _get_array_operator(self, ir_expr):
        ir_op = ir_expr.op
        if ir_op in ('unary', 'binop'):
            return ir_expr.fn
        elif ir_op == 'call':
            return self.typemap[ir_expr.func.name].typing_key
        raise NotImplementedError("Don't know how to find the operator for '{0}' expressions.".format(ir_op))

    def _get_operands(self, ir_expr):
        """Given a Numba IR expression, return the operands to the expression
        in order they appear in the expression.
        """
        ir_op = ir_expr.op
        if ir_op == 'binop':
            return (ir_expr.lhs, ir_expr.rhs)
        elif ir_op == 'unary':
            return ir_expr.list_vars()
        elif ir_op == 'call':
            return ir_expr.args
        raise NotImplementedError("Don't know how to find the operands for '{0}' expressions.".format(ir_op))

    def _translate_expr(self, ir_expr):
        """Translate the given expression from Numba IR to an array expression
        tree.
        """
        ir_op = ir_expr.op
        if ir_op == 'arrayexpr':
            return ir_expr.expr
        operands_or_args = [self.const_assigns.get(op_var.name, op_var) for op_var in self._get_operands(ir_expr)]
        return (self._get_array_operator(ir_expr), operands_or_args)

    def _handle_matches(self):
        """Iterate over the matches, trying to find which instructions should
        be rewritten, deleted, or moved.
        """
        replace_map = {}
        dead_vars = set()
        used_vars = defaultdict(int)
        for instr in self.array_assigns.values():
            expr = instr.value
            arr_inps = []
            arr_expr = (self._get_array_operator(expr), arr_inps)
            new_expr = ir.Expr(op='arrayexpr', loc=expr.loc, expr=arr_expr, ty=self.typemap[instr.target.name])
            new_instr = ir.Assign(new_expr, instr.target, instr.loc)
            replace_map[instr] = new_instr
            self.array_assigns[instr.target.name] = new_instr
            for operand in self._get_operands(expr):
                operand_name = operand.name
                if operand.is_temp and operand_name in self.array_assigns:
                    child_assign = self.array_assigns[operand_name]
                    child_expr = child_assign.value
                    child_operands = child_expr.list_vars()
                    for operand in child_operands:
                        used_vars[operand.name] += 1
                    arr_inps.append(self._translate_expr(child_expr))
                    if child_assign.target.is_temp:
                        dead_vars.add(child_assign.target.name)
                        replace_map[child_assign] = None
                elif operand_name in self.const_assigns:
                    arr_inps.append(self.const_assigns[operand_name])
                else:
                    used_vars[operand.name] += 1
                    arr_inps.append(operand)
        return (replace_map, dead_vars, used_vars)

    def _get_final_replacement(self, replacement_map, instr):
        """Find the final replacement instruction for a given initial
        instruction by chasing instructions in a map from instructions
        to replacement instructions.
        """
        replacement = replacement_map[instr]
        while replacement in replacement_map:
            replacement = replacement_map[replacement]
        return replacement

    def apply(self):
        """When we've found array expressions in a basic block, rewrite that
        block, returning a new, transformed block.
        """
        replace_map, dead_vars, used_vars = self._handle_matches()
        result = self.crnt_block.copy()
        result.clear()
        delete_map = {}
        for instr in self.crnt_block.body:
            if isinstance(instr, ir.Assign):
                if instr in replace_map:
                    replacement = self._get_final_replacement(replace_map, instr)
                    if replacement:
                        result.append(replacement)
                        for var in replacement.value.list_vars():
                            var_name = var.name
                            if var_name in delete_map:
                                result.append(delete_map.pop(var_name))
                            if used_vars[var_name] > 0:
                                used_vars[var_name] -= 1
                else:
                    result.append(instr)
            elif isinstance(instr, ir.Del):
                instr_value = instr.value
                if used_vars[instr_value] > 0:
                    used_vars[instr_value] -= 1
                    delete_map[instr_value] = instr
                elif instr_value not in dead_vars:
                    result.append(instr)
            else:
                result.append(instr)
        if delete_map:
            for instr in delete_map.values():
                result.insert_before_terminator(instr)
        return result