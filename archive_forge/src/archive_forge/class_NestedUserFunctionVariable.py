import functools
import inspect
import itertools
import types
from typing import Dict, List
import torch
from .. import variables
from ..bytecode_transformation import create_call_function, create_rot_n
from ..exc import unimplemented, Unsupported
from ..source import AttrSource, ConstantSource, DefaultsSource, GetItemSource
from ..utils import make_cell
from .base import typestr, VariableTracker
class NestedUserFunctionVariable(BaseUserFunctionVariable):
    _nonvar_fields = {'closure_scope', 'f_globals', *BaseUserFunctionVariable._nonvar_fields}

    def __init__(self, fn_name, code, f_globals, defaults, kwdefaults, annotations, closure, closure_scope, wrapped_reconstructible=None, **kwargs):
        super().__init__(**kwargs)
        assert isinstance(fn_name.as_python_constant(), str)
        assert isinstance(code.as_python_constant(), types.CodeType)
        assert isinstance(f_globals, dict)
        self.fn_name = fn_name
        self.code = code
        self.f_globals = f_globals
        self.defaults = defaults
        self.kwdefaults = kwdefaults
        self.annotations = annotations
        self.closure = closure
        if closure is None:
            closure_scope = None
        self.closure_scope = closure_scope
        self.wrapped_reconstructible: Optional[Union[Source, VariableTracker]] = wrapped_reconstructible

    def self_args(self):
        return []

    def get_code(self):
        return self.code.as_python_constant()

    def get_function(self):
        if self.closure:
            raise NotImplementedError()
        func = types.FunctionType(self.code.as_python_constant(), self.f_globals, self.fn_name.as_python_constant())
        if self.defaults:
            func.__defaults__ = self.defaults.as_python_constant()
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.as_python_constant()
        if self.annotations:
            annotations = self.annotations.as_python_constant()
            if isinstance(annotations, tuple):
                from itertools import pairwise
                annotations = dict(pairwise(annotations))
            assert isinstance(annotations, dict)
            func.__annotations__ = annotations
        return func

    def has_closure(self):
        return self.closure is not None

    def has_self(self):
        return False

    def get_globals(self):
        return self.f_globals

    def bind_args(self, parent, args, kwargs):
        from .misc import InlinedClosureVariable
        code = self.get_code()
        func = types.FunctionType(code, self.f_globals, self.fn_name.as_python_constant(), tuple(self.defaults.items) if self.defaults else None, tuple((make_cell(None) for _ in range(len(self.get_code().co_freevars)))))
        if self.kwdefaults:
            func.__kwdefaults__ = self.kwdefaults.items
        bound = inspect.signature(func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())
        wrap_args_kwargs(parent.output.root_tx, result)
        closure_cells = init_cellvars(parent, result, code)
        for idx, name in enumerate(code.co_freevars):
            cell = self.closure.items[idx]
            assert getattr(cell, name, name) == name
            assert name not in result
            if isinstance(cell, InlinedClosureVariable):
                cand = parent
                while cand and name not in cand.symbolic_locals:
                    cand = cand.parent
                if cand is None:
                    raise RuntimeError(f"Couldn't find {name} in the symbolic_locals of the inline interpreter stack")
                result[name] = cand.symbolic_locals[name]
            else:
                closure_cells[name] = self.closure.items[idx]
        return (result, closure_cells)

    def export_freevars(self, parent, child):
        code = self.get_code()
        for var in code.co_freevars:
            if var in child.symbolic_locals:
                parent.symbolic_locals[var] = child.symbolic_locals[var]

    def reconstruct(self, codegen):
        codegen.load_import_from(__name__, '_create_nested_fn')
        codegen(self.code)
        codegen.extend_output([codegen._create_load_const(self.f_globals)])
        codegen(self.fn_name)
        if self.defaults:
            codegen(self.defaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])
        if self.closure:
            codegen(self.closure)
        else:
            codegen.extend_output([codegen.create_load_const(None)])
        if self.kwdefaults:
            codegen(self.kwdefaults)
        else:
            codegen.extend_output([codegen.create_load_const(None)])
        if self.annotations:
            try:
                if isinstance(self.annotations, variables.ConstDictVariable):
                    annotations = {k: v.as_python_constant() for k, v in self.annotations.items.items()}
                else:
                    annotations = tuple([v.as_python_constant() for v in self.annotations.items])
                codegen.extend_output([codegen._create_load_const(annotations)])
            except NotImplementedError:
                codegen(self.annotations)
        else:
            codegen.extend_output([codegen.create_load_const(None)])
        codegen.extend_output(create_call_function(7, push_null=True))
        if self.wrapped_reconstructible:
            codegen.load_import_from('functools', 'wraps')
            codegen(self.wrapped_reconstructible)
            codegen.extend_output(create_call_function(1, True))
            codegen.extend_output(create_rot_n(2))
            codegen.extend_output(create_call_function(1, True))
        return []