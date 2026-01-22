import collections
import dataclasses
import functools
import inspect
import itertools
import operator
import sys
import types
from typing import Dict, List
import torch._C
import torch._numpy as tnp
from .. import config, polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, TypeSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
from .user_defined import UserDefinedObjectVariable
class SkipFilesVariable(VariableTracker):

    def __init__(self, value, reason=None, **kwargs):
        super().__init__(**kwargs)
        self.value = value
        self.reason = reason

    def python_type(self):
        return type(self.value)

    def as_python_constant(self):
        return self.value

    @classmethod
    def create_with_source(cls, value, source):
        install_guard(source.make_guard(GuardBuilder.FUNCTION_MATCH))
        return cls(value, source=source)

    @staticmethod
    @functools.lru_cache(None)
    def fold_through_function_to_wrapper():
        return {collections.namedtuple: variables.UserDefinedClassVariable}

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from .builtin import BuiltinVariable
        if inspect.getattr_static(self.value, '_torchdynamo_disable', False):
            unimplemented(f'call torch._dynamo.disable() wrapped function {self.value}')
        elif self.value is collections.OrderedDict:
            return BuiltinVariable.call_custom_dict(tx, collections.OrderedDict, *args, **kwargs)
        elif self.value is collections.defaultdict and len(args) <= 1 and DefaultDictVariable.is_supported_arg(args[0]):
            return DefaultDictVariable({}, collections.defaultdict, args[0], mutable_local=MutableLocal())
        elif self.value in self.fold_through_function_to_wrapper().keys() and check_constant_args(args, kwargs):
            value = self.value(*[x.as_python_constant() for x in args], **{k: v.as_python_constant() for k, v in kwargs.items()})
            return self.fold_through_function_to_wrapper().get(self.value)(value, mutable_local=MutableLocal())
        elif self.value is itertools.product and (not kwargs) and all((arg.has_unpack_var_sequence(tx) for arg in args)):
            seqs = [arg.unpack_var_sequence(tx) for arg in args]
            items = []
            for item in itertools.product(*seqs):
                items.append(variables.TupleVariable(list(item)))
            return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
        elif self.value is itertools.chain and (not kwargs) and all((arg.has_unpack_var_sequence(tx) for arg in args)):
            seqs = [arg.unpack_var_sequence(tx) for arg in args]
            items = []
            for item in itertools.chain(*seqs):
                items.append(item)
            return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
        elif self.value is itertools.accumulate:
            from .builtin import BuiltinVariable
            if any((key not in ['initial', 'func'] for key in kwargs.keys())):
                unimplemented(f'Unsupported kwargs for itertools.accumulate: {','.join(set(kwargs.keys()) - {'initial', 'func'})}')
            acc = kwargs.get('initial')
            if len(args) in [1, 2] and args[0].has_unpack_var_sequence(tx):
                seq = args[0].unpack_var_sequence(tx)
                if 'func' in kwargs and len(args) == 1:
                    func = kwargs['func'].call_function
                elif len(args) == 2:
                    func = args[1].call_function
                elif len(args) == 1:
                    func = BuiltinVariable(operator.add).call_function
                else:
                    unimplemented('itertools.accumulate can only accept one of: `func` kwarg, pos 2 arg')
            else:
                unimplemented('Unsupported arguments for itertools.accumulate')
            items = []
            if acc is not None:
                items.append(acc)
            for item in seq:
                if acc is None:
                    acc = item
                else:
                    try:
                        acc = func(tx, [acc, item], {})
                    except Exception:
                        raise unimplemented(f'Unexpected failure in invoking function during accumulate. Failed running func {func}({item}{acc})')
                items.append(acc)
            return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
        elif self.value is itertools.combinations and (not kwargs) and (len(args) == 2) and args[0].has_unpack_var_sequence(tx) and args[1].is_python_constant():
            iterable = args[0].unpack_var_sequence(tx)
            r = args[1].as_python_constant()
            items = []
            for item in itertools.combinations(iterable, r):
                items.append(variables.TupleVariable(list(item)))
            return variables.ListIteratorVariable(items, mutable_local=MutableLocal())
        elif self.value is itertools.groupby:
            if any((kw != 'key' for kw in kwargs.keys())):
                unimplemented(f'Unsupported kwargs for itertools.groupby: {','.join(set(kwargs.keys()) - {'key'})}')

            def retrieve_const_key(key):
                if isinstance(key, variables.SymNodeVariable):
                    return key.evaluate_expr()
                elif isinstance(key, variables.ConstantVariable):
                    return key.as_python_constant()
                else:
                    raise unimplemented('Unsupported key type for itertools.groupby: ' + str(type(key)))
            if len(args) == 1 and args[0].has_unpack_var_sequence(tx):
                seq = args[0].unpack_var_sequence(tx)
                keyfunc = (lambda x: retrieve_const_key(kwargs.get('key').call_function(tx, [x], {}))) if 'key' in kwargs else None
            else:
                unimplemented('Unsupported arguments for itertools.groupby')
            result = []
            try:
                for k, v in itertools.groupby(seq, key=keyfunc):
                    result.append(variables.TupleVariable([variables.ConstantVariable.create(k) if variables.ConstantVariable.is_literal(k) else k, variables.ListIteratorVariable(list(v), mutable_local=MutableLocal())], mutable_local=MutableLocal()))
            except Exception:
                raise unimplemented('Unexpected failure when calling itertools.groupby')
            return variables.ListIteratorVariable(result, mutable_local=MutableLocal())
        elif self.value is functools.wraps and (not kwargs) and (len(args) == 1) and (args[0].source is not None or args[0].can_reconstruct(tx.output.root_tx)):

            def wraps(fn):
                if isinstance(fn, variables.NestedUserFunctionVariable):
                    if args[0].source:
                        reconstructible = args[0].source
                    else:
                        reconstructible = args[0]
                    return fn.clone(wrapped_reconstructible=reconstructible)
                unimplemented(f'functools.wraps({fn})')
            return variables.LambdaVariable(wraps)
        elif self.value is collections.deque and (not kwargs):
            if len(args) == 0:
                items = []
            elif len(args) == 1 and args[0].has_unpack_var_sequence(tx):
                items = args[0].unpack_var_sequence(tx)
            else:
                unimplemented('deque() with more than 1 arg not supported')
            return variables.lists.DequeVariable(items, mutable_local=MutableLocal())
        elif self.value is functools.partial:
            if not args:
                unimplemented('functools.partial malformed')
            fn = args[0]
            rest_args = args[1:]
            return variables.functions.FunctoolsPartialVariable(fn, args=rest_args, keywords=kwargs)
        elif self.value is itertools.repeat:
            if len(args) < 2:
                return variables.RepeatIteratorVariable(*args, mutable_local=MutableLocal())
            from .builder import SourcelessBuilder
            return tx.inline_user_function_return(SourcelessBuilder()(tx, polyfill.repeat), args, kwargs)
        elif self.value is itertools.count:
            return variables.CountIteratorVariable(*args, mutable_local=MutableLocal())
        elif self.value is itertools.cycle:
            return variables.CycleIteratorVariable(*args, mutable_local=MutableLocal())
        else:
            try:
                path = inspect.getfile(self.value)
            except TypeError:
                path = f'Builtin {self.value.__name__}'
            msg = f"'skip function {self.value.__qualname__} in file {path}'"
            msg += f"', {self.reason}'" if self.reason else ''
            unimplemented(msg)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if self.value in {collections.OrderedDict, collections.defaultdict} and name == 'fromkeys':
            from .builtin import BuiltinVariable
            return BuiltinVariable.call_custom_dict_fromkeys(tx, self.value, *args, **kwargs)
        return super().call_method(tx, name, args, kwargs)