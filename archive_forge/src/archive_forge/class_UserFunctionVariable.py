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
class UserFunctionVariable(BaseUserFunctionVariable):
    """Some unsupported user-defined global function"""

    def __init__(self, fn, is_constant=False, **kwargs):
        super().__init__(**kwargs)
        if getattr(fn, '_dynamo_marked_constant', False):
            self.is_constant = True
        else:
            self.is_constant = False
        assert isinstance(fn, (types.FunctionType, torch.jit.ScriptFunction)), f'expected FunctionType found {typestr(fn)} {fn}'
        fn = inspect.getattr_static(fn, '_torchdynamo_inline', fn)
        if inspect.getattr_static(fn, '__script_if_tracing_wrapper', False):
            fn = inspect.getattr_static(fn, '__original_fn', fn)
        self.fn: types.FunctionType = fn

    def self_args(self):
        return []

    def get_function(self):
        return self.fn

    def get_code(self):
        return self.fn.__code__

    def python_type(self):
        return types.FunctionType

    def has_self(self):
        return getattr(self.fn, '__self__', None) is not None

    def get_globals(self):
        return self.fn.__globals__

    def bind_args(self, parent, args, kwargs):
        assert not self.is_constant
        tx = parent.output.root_tx
        wrap = functools.partial(wrap_bound_arg, tx=tx)
        fn: types.FunctionType = self.fn
        defaults = fn.__defaults__ or []
        defaults_sources = [None if self.source is None else DefaultsSource(self.source, idx) for idx, _ in enumerate(defaults)]
        fake_func = types.FunctionType(fn.__code__, fn.__globals__, fn.__name__, tuple([wrap(val=arg, source=source) for arg, source in zip(defaults, defaults_sources)]), fn.__closure__)
        if fn.__kwdefaults__:
            kwdefaults_sources = {k: None if self.source is None else DefaultsSource(self.source, k, is_kw=True) for k in fn.__kwdefaults__}
            fake_func.__kwdefaults__ = {k: wrap(val=v, source=kwdefaults_sources[k]) for k, v in fn.__kwdefaults__.items()}
        bound = inspect.signature(fake_func).bind(*args, **kwargs)
        bound.apply_defaults()
        result = dict(bound.arguments.items())
        wrap_args_kwargs(tx, result)
        closure_cells = init_cellvars(parent, result, fn.__code__)
        closure = self.fn.__closure__ or ()
        assert len(closure) == len(self.fn.__code__.co_freevars)
        for idx, name, cell in zip(itertools.count(), self.fn.__code__.co_freevars, closure):
            if name == '__class__':
                source = AttrSource(self.source, '__class__') if self.source else None
                result[name] = variables.UserDefinedClassVariable(cell.cell_contents, source=source)
            else:
                var = tx.match_nested_cell(name, cell)
                if var is not None:
                    result[name] = var
                elif self.source:
                    from .builder import VariableBuilder
                    side_effects = parent.output.side_effects
                    if cell in side_effects:
                        out = side_effects[cell]
                    else:
                        closure_cell = GetItemSource(AttrSource(self.source, '__closure__'), idx)
                        closure_cell_contents = AttrSource(closure_cell, 'cell_contents')
                        contents_var = VariableBuilder(parent, closure_cell_contents)(cell.cell_contents)
                        if closure_cell_contents.name() not in tx.mutated_closure_cell_contents:
                            result[name] = contents_var
                            continue
                        out = side_effects.track_cell_existing(closure_cell, cell)
                        side_effects.store_cell(out, contents_var)
                    result[name] = out
                else:
                    from .builder import SourcelessBuilder
                    result[name] = SourcelessBuilder()(tx, cell.cell_contents)
        return (result, closure_cells)

    def export_freevars(self, parent, child):
        pass

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if self.is_constant:
            return invoke_and_store_as_constant(tx, self.fn, self.get_name(), args, kwargs)
        return super().call_function(tx, args, kwargs)