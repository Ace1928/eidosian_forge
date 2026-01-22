from __future__ import annotations
import argparse
import collections
from functools import update_wrapper
import inspect
import itertools
import operator
import os
import re
import sys
from typing import TYPE_CHECKING
import uuid
import pytest
class PytestFixtureFunctions(plugin_base.FixtureFunctions):

    def skip_test_exception(self, *arg, **kw):
        return pytest.skip.Exception(*arg, **kw)

    @property
    def add_to_marker(self):
        return pytest.mark

    def mark_base_test_class(self):
        return pytest.mark.usefixtures('setup_class_methods', 'setup_test_methods')
    _combination_id_fns = {'i': lambda obj: obj, 'r': repr, 's': str, 'n': lambda obj: obj.__name__ if hasattr(obj, '__name__') else type(obj).__name__}

    def combinations(self, *arg_sets, **kw):
        """Facade for pytest.mark.parametrize.

        Automatically derives argument names from the callable which in our
        case is always a method on a class with positional arguments.

        ids for parameter sets are derived using an optional template.

        """
        from sqlalchemy.testing import exclusions
        if len(arg_sets) == 1 and hasattr(arg_sets[0], '__next__'):
            arg_sets = list(arg_sets[0])
        argnames = kw.pop('argnames', None)

        def _filter_exclusions(args):
            result = []
            gathered_exclusions = []
            for a in args:
                if isinstance(a, exclusions.compound):
                    gathered_exclusions.append(a)
                else:
                    result.append(a)
            return (result, gathered_exclusions)
        id_ = kw.pop('id_', None)
        tobuild_pytest_params = []
        has_exclusions = False
        if id_:
            _combination_id_fns = self._combination_id_fns
            _arg_getter = operator.itemgetter(0, *[idx for idx, char in enumerate(id_) if char in ('n', 'r', 's', 'a')])
            fns = [(operator.itemgetter(idx), _combination_id_fns[char]) for idx, char in enumerate(id_) if char in _combination_id_fns]
            for arg in arg_sets:
                if not isinstance(arg, tuple):
                    arg = (arg,)
                fn_params, param_exclusions = _filter_exclusions(arg)
                parameters = _arg_getter(fn_params)[1:]
                if param_exclusions:
                    has_exclusions = True
                tobuild_pytest_params.append((parameters, param_exclusions, '-'.join((comb_fn(getter(arg)) for getter, comb_fn in fns))))
        else:
            for arg in arg_sets:
                if not isinstance(arg, tuple):
                    arg = (arg,)
                fn_params, param_exclusions = _filter_exclusions(arg)
                if param_exclusions:
                    has_exclusions = True
                tobuild_pytest_params.append((fn_params, param_exclusions, None))
        pytest_params = []
        for parameters, param_exclusions, id_ in tobuild_pytest_params:
            if has_exclusions:
                parameters += (param_exclusions,)
            param = pytest.param(*parameters, id=id_)
            pytest_params.append(param)

        def decorate(fn):
            if inspect.isclass(fn):
                if has_exclusions:
                    raise NotImplementedError('exclusions not supported for class level combinations')
                if '_sa_parametrize' not in fn.__dict__:
                    fn._sa_parametrize = []
                fn._sa_parametrize.append((argnames, pytest_params))
                return fn
            else:
                _fn_argnames = inspect.getfullargspec(fn).args[1:]
                if argnames is None:
                    _argnames = _fn_argnames
                else:
                    _argnames = re.split(', *', argnames)
                if has_exclusions:
                    existing_exl = sum((1 for n in _fn_argnames if n.startswith('_exclusions')))
                    current_exclusion_name = f'_exclusions_{existing_exl}'
                    _argnames += [current_exclusion_name]

                    @_pytest_fn_decorator
                    def check_exclusions(fn, *args, **kw):
                        _exclusions = args[-1]
                        if _exclusions:
                            exlu = exclusions.compound().add(*_exclusions)
                            fn = exlu(fn)
                        return fn(*args[:-1], **kw)
                    fn = check_exclusions(fn, add_positional_parameters=(current_exclusion_name,))
                return pytest.mark.parametrize(_argnames, pytest_params)(fn)
        return decorate

    def param_ident(self, *parameters):
        ident = parameters[0]
        return pytest.param(*parameters[1:], id=ident)

    def fixture(self, *arg, **kw):
        from sqlalchemy.testing import config
        from sqlalchemy.testing import asyncio
        if len(arg) > 0 and callable(arg[0]):
            fn = arg[0]
            arg = arg[1:]
        else:
            fn = None
        fixture = pytest.fixture(*arg, **kw)

        def wrap(fn):
            if config.any_async:
                fn = asyncio._maybe_async_wrapper(fn)
            fn = fixture(fn)
            return fn
        if fn:
            return wrap(fn)
        else:
            return wrap

    def get_current_test_name(self):
        return os.environ.get('PYTEST_CURRENT_TEST')

    def async_test(self, fn):
        from sqlalchemy.testing import asyncio

        @_pytest_fn_decorator
        def decorate(fn, *args, **kwargs):
            asyncio._run_coroutine_function(fn, *args, **kwargs)
        return decorate(fn)