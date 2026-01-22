import base64
import cloudpickle
from copy import deepcopy
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
import numpy as np
from triad import assert_or_throw, to_uuid
from triad.utils.convert import get_full_type_path
from tune._utils import product
from tune._utils.math import (
class TuningParametersTemplate:
    """Parameter template to extract tuning parameter expressions from
    nested data structure

    :param raw: the dictionary of input parameters.

    .. note::

        Please use :func:`~tune.concepts.space.parameters.to_template`
        to initialize this class.

    .. code-block:: python

        # common cases
        to_template(dict(a=1, b=1))
        to_template(dict(a=Rand(0, 1), b=1))

        # expressions may nest in dicts or arrays
        template = to_template(
            dict(a=dict(x1=Rand(0, 1), x2=Rand(3,4)), b=[Grid("a", "b")]))

        assert [Rand(0, 1), Rand(3, 4), Grid("a", "b")] == template.params
        assert dict(
            p0=Rand(0, 1), p1=Rand(3, 4), p2=Grid("a", "b")
        ) == template.params_dict
        assert dict(a=1, x2=3), b=["a"]) == template.fill([1, 3, "a"])
        assert dict(a=1, x2=3), b=["a"]) == template.fill_dict(
            dict(p2="a", p1=3, p0=1)
        )

    """

    def __init__(self, raw: Dict[str, Any]):
        self._units: List[_MapUnit] = []
        self._has_grid = False
        self._has_stochastic = False
        self._func_positions: List[List[Any]] = []
        self._template: Dict[str, Any] = self._copy(raw, [], {})
        self._uuid = ''

    def __eq__(self, other: Any) -> bool:
        """Check if the other object represents the same template

        :param other: an object convertible to ``TuningParametersTemplate``
            by :func:`~tune.concepts.space.parameters.to_template`
        :return: whether they are the same
        """
        o = to_template(other)
        return self._has_grid == o._has_grid and self._has_stochastic == o._has_stochastic and (self._template == o._template) and (self._units == o._units) and (self._func_positions == o._func_positions)

    def __uuid__(self):
        """The unique id representing this template"""
        if self._uuid == '':
            self._uuid = to_uuid(self._units, self._template)
        return self._uuid

    def __repr__(self) -> str:
        return repr(self.fill([x.expr for x in self._units]))

    @property
    def template(self) -> Dict[str, Any]:
        """The template dictionary, all tuning
        expressions will be replaced by ``None``
        """
        return self._template

    @property
    def simple_value(self) -> Dict[str, Any]:
        """If the template contains no tuning expression, it's simple
        and it will return parameters dictionary, otherwise, ``ValueError``
        will be raised
        """
        assert_or_throw(self.empty, ValueError('template contains tuning expressions'))
        if len(self._func_positions) == 0:
            return self._template
        return self._fill_funcs(deepcopy(self._template))

    @property
    def empty(self) -> bool:
        """Whether the template contains any tuning expression"""
        return len(self._units) == 0

    @property
    def has_grid(self) -> bool:
        """Whether the template contains grid expressions"""
        return self._has_grid

    @property
    def has_stochastic(self) -> bool:
        """Whether the template contains stochastic expressions"""
        return self._has_stochastic

    @property
    def params(self) -> List[TuningParameterExpression]:
        """Get all tuning parameter expressions in depth-first order"""
        return [x.expr for x in self._units]

    @property
    def params_dict(self) -> Dict[str, TuningParameterExpression]:
        """Get all tuning parameter expressions in depth-first order,
        with correspondent made-up new keys p0, p1, p2, ...
        """
        return {f'p{i}': x for i, x in enumerate(self.params)}

    def fill(self, params: List[Any]) -> Dict[str, Any]:
        """Fill the original data structure with values

        :param params: the list of values to be filled into the original
          data structure, in depth-first order
        :param copy: whether to return a deeply copied paramters,
          defaults to False
        :return: the original data structure filled with values
        """
        assert_or_throw(len(self._units) == len(params), ValueError('params count does not match template requirment'))
        template = deepcopy(self._template)
        i = 0
        for u in self._units:
            for path in u.positions:
                self._fill_path(template, path, params[i])
            i += 1
        return self._fill_funcs(template)

    def fill_dict(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Fill the original data structure with dictionary of values

        :param params: the dictionary of values to be filled into the original
          data structure, keys must be p0, p1, p2, ...
        :param copy: whether to return a deeply copied paramters,
          defaults to False
        :return: the original data structure filled with values
        """
        temp = [params[f'p{i}'] for i in range(len(params))]
        return self.fill(temp)

    def encode(self) -> str:
        """Convert the template to a base64 string"""
        return base64.b64encode(cloudpickle.dumps(self)).decode('ascii')

    @staticmethod
    def decode(data: str) -> 'TuningParametersTemplate':
        """Retrieve the template from a base64 string"""
        return cloudpickle.loads(base64.b64decode(data.encode('ascii')))

    def product_grid(self) -> Iterable['TuningParametersTemplate']:
        """cross product all grid parameters

        :yield: new templates with the grid paramters filled

        .. code-block:: python

            assert [dict(a=1,b=Rand(0,1)), dict(a=2,b=Rand(0,1))] ==                 list(to_template(dict(a=Grid(1,2),b=Rand(0,1))).product_grid())
        """
        if not self.has_grid:
            yield self
        else:
            gu: List[Tuple[int, List[Any]]] = [(i, list(u.expr)) for i, u in enumerate(self._units) if isinstance(u.expr, Grid)]
            yield from self._partial_fill([x[0] for x in gu], product([data for _, data in gu], safe=True, remove_empty=True))

    def sample(self, n: int, seed: Any=None) -> Iterable['TuningParametersTemplate']:
        """sample all stochastic parameters

        :param n: number of samples, must be a positive integer
        :param seed: random seed defaulting to None.
          It will take effect if it is not None.

        :yield: new templates with the grid paramters filled

        .. code-block:: python

            assert [dict(a=1.1,b=Grid(0,1)), dict(a=1.5,b=Grid(0,1))] ==                 list(to_template(dict(a=Rand(1,2),b=Grid(0,1))).sample(2,0))
        """
        assert_or_throw(n > 0, ValueError('sample count must be positive'))
        if not self.has_stochastic:
            yield self
        else:
            if seed is not None:
                np.random.seed(seed)
            gu: List[Tuple[int, List[Any]]] = [(i, u.expr.generate_many(n)) for i, u in enumerate(self._units) if isinstance(u.expr, StochasticExpression)]
            yield from self._partial_fill([x[0] for x in gu], zip(*[data for _, data in gu]))

    def concat(self, other: 'TuningParametersTemplate') -> 'TuningParametersTemplate':
        """Concatenate with another template and generate a new template.

        .. note::

            The other template must not have any key existed in this template, otherwise
            ``ValueError`` will be raised

        :return: the merged template
        """
        res = TuningParametersTemplate({})
        res._units = [x.copy() for x in self._units]
        res._has_grid = self._has_grid | other._has_grid
        res._has_stochastic = self._has_stochastic | other._has_stochastic
        res._template = dict(self._template)
        res._func_positions = self._func_positions + other._func_positions
        for k, v in other._template.items():
            assert_or_throw(k not in res._template, ValueError(f'{k} already exists in the original template'))
            res._template[k] = v
        if not other.empty:
            temp_map = {id(x.expr): x for x in res._units}
            for u in other._units:
                if id(u.expr) in temp_map:
                    temp_map[id(u.expr)].positions += u.positions
                else:
                    res._units.append(u.copy())
        return res

    def _fill_funcs(self, obj: Dict[str, Any]) -> Dict[str, Any]:

        def realize_func(path: List[Any]) -> None:
            r: Any = obj
            for p in path[:-1]:
                r = r[p]
            r[path[-1]] = r[path[-1]]()
        for path in self._func_positions:
            realize_func(path)
        return obj

    def _copy(self, src: Any, keys: List[Any], idx: Dict[int, _MapUnit]) -> Any:
        if isinstance(src, dict):
            ddest: Dict[str, Any] = {}
            for k, v in src.items():
                nk = keys + [k]
                if isinstance(v, TuningParameterExpression):
                    ddest[k] = None
                    self._add(nk, v, idx)
                else:
                    ddest[k] = self._copy(v, nk, idx)
            return ddest
        elif isinstance(src, list):
            adest: List[Any] = []
            for i in range(len(src)):
                nk = keys + [i]
                if isinstance(src[i], TuningParameterExpression):
                    adest.append(None)
                    self._add(nk, src[i], idx)
                else:
                    adest.append(self._copy(src[i], nk, idx))
            return adest
        elif isinstance(src, FuncParam):
            self._func_positions.append(keys)
            args = self._copy(src._args, keys, idx)
            kwargs = self._copy(src._kwargs, keys, idx)
            return FuncParam(src._func, *args, **kwargs)
        else:
            return src

    def _add(self, keys: List[Any], expr: TuningParameterExpression, idx: Dict[int, _MapUnit]) -> None:
        if id(expr) not in idx:
            mu = _MapUnit(expr)
            self._units.append(mu)
            idx[id(expr)] = mu
        else:
            mu = idx[id(expr)]
        mu.positions.append(keys)
        if isinstance(expr, Grid):
            self._has_grid = True
        else:
            self._has_stochastic = True

    def _fill_path(self, root: Dict[str, Any], path: List[Any], v: Any) -> None:
        r = root
        for p in path[:-1]:
            r = r[p]
        r[path[-1]] = v

    def _partial_fill(self, idx: List[int], params_list: Iterable[List[Any]]) -> Iterable['TuningParametersTemplate']:
        new_units = [u for i, u in enumerate(self._units) if i not in idx]
        has_grid = any((isinstance(x.expr, Grid) for x in new_units))
        has_stochastic = any((isinstance(x.expr, StochasticExpression) for x in new_units))
        for params in params_list:
            new_template = deepcopy(self._template)
            for pi, i in enumerate(idx):
                for path in self._units[i].positions:
                    self._fill_path(new_template, path, params[pi])
            t = TuningParametersTemplate({})
            t._units = new_units
            t._template = new_template
            t._has_grid = has_grid
            t._has_stochastic = has_stochastic
            t._func_positions = self._func_positions
            if t.empty and len(t._func_positions) > 0:
                t._template = t._fill_funcs(t._template)
                t._func_positions = []
            yield t