from __future__ import annotations
from typing import Any
from functools import reduce
from itertools import permutations
from sympy.combinatorics import Permutation
from sympy.core import (
from sympy.core.cache import cacheit
from sympy.core.symbol import Symbol, Dummy
from sympy.core.symbol import Str
from sympy.core.sympify import _sympify
from sympy.functions import factorial
from sympy.matrices import ImmutableDenseMatrix as Matrix
from sympy.solvers import solve
from sympy.utilities.exceptions import (sympy_deprecation_warning,
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.simplify.simplify import simplify
class CoordSystem(Basic):
    """
    A coordinate system defined on the patch.

    Explanation
    ===========

    Coordinate system is a system that uses one or more coordinates to uniquely
    determine the position of the points or other geometric elements on a
    manifold [1].

    By passing ``Symbols`` to *symbols* parameter, user can define the name and
    assumptions of coordinate symbols of the coordinate system. If not passed,
    these symbols are generated automatically and are assumed to be real valued.

    By passing *relations* parameter, user can define the transform relations of
    coordinate systems. Inverse transformation and indirect transformation can
    be found automatically. If this parameter is not passed, coordinate
    transformation cannot be done.

    Parameters
    ==========

    name : str
        The name of the coordinate system.

    patch : Patch
        The patch where the coordinate system is defined.

    symbols : list of Symbols, optional
        Defines the names and assumptions of coordinate symbols.

    relations : dict, optional
        Key is a tuple of two strings, who are the names of the systems where
        the coordinates transform from and transform to.
        Value is a tuple of the symbols before transformation and a tuple of
        the expressions after transformation.

    Examples
    ========

    We define two-dimensional Cartesian coordinate system and polar coordinate
    system.

    >>> from sympy import symbols, pi, sqrt, atan2, cos, sin
    >>> from sympy.diffgeom import Manifold, Patch, CoordSystem
    >>> m = Manifold('M', 2)
    >>> p = Patch('P', m)
    >>> x, y = symbols('x y', real=True)
    >>> r, theta = symbols('r theta', nonnegative=True)
    >>> relation_dict = {
    ... ('Car2D', 'Pol'): [(x, y), (sqrt(x**2 + y**2), atan2(y, x))],
    ... ('Pol', 'Car2D'): [(r, theta), (r*cos(theta), r*sin(theta))]
    ... }
    >>> Car2D = CoordSystem('Car2D', p, (x, y), relation_dict)
    >>> Pol = CoordSystem('Pol', p, (r, theta), relation_dict)

    ``symbols`` property returns ``CoordinateSymbol`` instances. These symbols
    are not same with the symbols used to construct the coordinate system.

    >>> Car2D
    Car2D
    >>> Car2D.dim
    2
    >>> Car2D.symbols
    (x, y)
    >>> _[0].func
    <class 'sympy.diffgeom.diffgeom.CoordinateSymbol'>

    ``transformation()`` method returns the transformation function from
    one coordinate system to another. ``transform()`` method returns the
    transformed coordinates.

    >>> Car2D.transformation(Pol)
    Lambda((x, y), Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]]))
    >>> Car2D.transform(Pol)
    Matrix([
    [sqrt(x**2 + y**2)],
    [      atan2(y, x)]])
    >>> Car2D.transform(Pol, [1, 2])
    Matrix([
    [sqrt(5)],
    [atan(2)]])

    ``jacobian()`` method returns the Jacobian matrix of coordinate
    transformation between two systems. ``jacobian_determinant()`` method
    returns the Jacobian determinant of coordinate transformation between two
    systems.

    >>> Pol.jacobian(Car2D)
    Matrix([
    [cos(theta), -r*sin(theta)],
    [sin(theta),  r*cos(theta)]])
    >>> Pol.jacobian(Car2D, [1, pi/2])
    Matrix([
    [0, -1],
    [1,  0]])
    >>> Car2D.jacobian_determinant(Pol)
    1/sqrt(x**2 + y**2)
    >>> Car2D.jacobian_determinant(Pol, [1,0])
    1

    References
    ==========

    .. [1] https://en.wikipedia.org/wiki/Coordinate_system

    """

    def __new__(cls, name, patch, symbols=None, relations={}, **kwargs):
        if not isinstance(name, Str):
            name = Str(name)
        if symbols is None:
            names = kwargs.get('names', None)
            if names is None:
                symbols = Tuple(*[Symbol('%s_%s' % (name.name, i), real=True) for i in range(patch.dim)])
            else:
                sympy_deprecation_warning(f"\nThe 'names' argument to CoordSystem is deprecated. Use 'symbols' instead. That\nis, replace\n\n    CoordSystem(..., names={names})\n\nwith\n\n    CoordSystem(..., symbols=[{', '.join(['Symbol(' + repr(n) + ', real=True)' for n in names])}])\n                    ", deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
                symbols = Tuple(*[Symbol(n, real=True) for n in names])
        else:
            syms = []
            for s in symbols:
                if isinstance(s, Symbol):
                    syms.append(Symbol(s.name, **s._assumptions.generator))
                elif isinstance(s, str):
                    sympy_deprecation_warning(f'\n\nPassing a string as the coordinate symbol name to CoordSystem is deprecated.\nPass a Symbol with the appropriate name and assumptions instead.\n\nThat is, replace {s} with Symbol({s!r}, real=True).\n                        ', deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
                    syms.append(Symbol(s, real=True))
            symbols = Tuple(*syms)
        rel_temp = {}
        for k, v in relations.items():
            s1, s2 = k
            if not isinstance(s1, Str):
                s1 = Str(s1)
            if not isinstance(s2, Str):
                s2 = Str(s2)
            key = Tuple(s1, s2)
            if isinstance(v, Lambda):
                v = (tuple(v.signature), tuple(v.expr))
            else:
                v = (tuple(v[0]), tuple(v[1]))
            rel_temp[key] = v
        relations = Dict(rel_temp)
        obj = super().__new__(cls, name, patch, symbols, relations)
        obj.transforms = _deprecated_dict("\n            CoordSystem.transforms is deprecated. The CoordSystem class is now\n            immutable. Use the 'relations' keyword argument to the\n            CoordSystems() constructor to specify relations.\n            ", {})
        obj._names = [str(n) for n in symbols]
        obj.patch.coord_systems.append(obj)
        obj._dummies = [Dummy(str(n)) for n in symbols]
        obj._dummy = Dummy()
        return obj

    @property
    def name(self):
        return self.args[0]

    @property
    def patch(self):
        return self.args[1]

    @property
    def manifold(self):
        return self.patch.manifold

    @property
    def symbols(self):
        return tuple((CoordinateSymbol(self, i, **s._assumptions.generator) for i, s in enumerate(self.args[2])))

    @property
    def relations(self):
        return self.args[3]

    @property
    def dim(self):
        return self.patch.dim

    def transformation(self, sys):
        """
        Return coordinate transformation function from *self* to *sys*.

        Parameters
        ==========

        sys : CoordSystem

        Returns
        =======

        sympy.Lambda

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transformation(R2_p)
        Lambda((x, y), Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]]))

        """
        signature = self.args[2]
        key = Tuple(self.name, sys.name)
        if self == sys:
            expr = Matrix(self.symbols)
        elif key in self.relations:
            expr = Matrix(self.relations[key][1])
        elif key[::-1] in self.relations:
            expr = Matrix(self._inverse_transformation(sys, self))
        else:
            expr = Matrix(self._indirect_transformation(self, sys))
        return Lambda(signature, expr)

    @staticmethod
    def _solve_inverse(sym1, sym2, exprs, sys1_name, sys2_name):
        ret = solve([t[0] - t[1] for t in zip(sym2, exprs)], list(sym1), dict=True)
        if len(ret) == 0:
            temp = 'Cannot solve inverse relation from {} to {}.'
            raise NotImplementedError(temp.format(sys1_name, sys2_name))
        elif len(ret) > 1:
            temp = 'Obtained multiple inverse relation from {} to {}.'
            raise ValueError(temp.format(sys1_name, sys2_name))
        return ret[0]

    @classmethod
    def _inverse_transformation(cls, sys1, sys2):
        forward = sys1.transform(sys2)
        inv_results = cls._solve_inverse(sys1.symbols, sys2.symbols, forward, sys1.name, sys2.name)
        signature = tuple(sys1.symbols)
        return [inv_results[s] for s in signature]

    @classmethod
    @cacheit
    def _indirect_transformation(cls, sys1, sys2):
        rel = sys1.relations
        path = cls._dijkstra(sys1, sys2)
        transforms = []
        for s1, s2 in zip(path, path[1:]):
            if (s1, s2) in rel:
                transforms.append(rel[s1, s2])
            else:
                sym2, inv_exprs = rel[s2, s1]
                sym1 = tuple((Dummy() for i in sym2))
                ret = cls._solve_inverse(sym2, sym1, inv_exprs, s2, s1)
                ret = tuple((ret[s] for s in sym2))
                transforms.append((sym1, ret))
        syms = sys1.args[2]
        exprs = syms
        for newsyms, newexprs in transforms:
            exprs = tuple((e.subs(zip(newsyms, exprs)) for e in newexprs))
        return exprs

    @staticmethod
    def _dijkstra(sys1, sys2):
        relations = sys1.relations
        graph = {}
        for s1, s2 in relations.keys():
            if s1 not in graph:
                graph[s1] = {s2}
            else:
                graph[s1].add(s2)
            if s2 not in graph:
                graph[s2] = {s1}
            else:
                graph[s2].add(s1)
        path_dict = {sys: [0, [], 0] for sys in graph}

        def visit(sys):
            path_dict[sys][2] = 1
            for newsys in graph[sys]:
                distance = path_dict[sys][0] + 1
                if path_dict[newsys][0] >= distance or not path_dict[newsys][1]:
                    path_dict[newsys][0] = distance
                    path_dict[newsys][1] = list(path_dict[sys][1])
                    path_dict[newsys][1].append(sys)
        visit(sys1.name)
        while True:
            min_distance = max(path_dict.values(), key=lambda x: x[0])[0]
            newsys = None
            for sys, lst in path_dict.items():
                if 0 < lst[0] <= min_distance and (not lst[2]):
                    min_distance = lst[0]
                    newsys = sys
            if newsys is None:
                break
            visit(newsys)
        result = path_dict[sys2.name][1]
        result.append(sys2.name)
        if result == [sys2.name]:
            raise KeyError('Two coordinate systems are not connected.')
        return result

    def connect_to(self, to_sys, from_coords, to_exprs, inverse=True, fill_in_gaps=False):
        sympy_deprecation_warning("\n            The CoordSystem.connect_to() method is deprecated. Instead,\n            generate a new instance of CoordSystem with the 'relations'\n            keyword argument (CoordSystem classes are now immutable).\n            ", deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
        from_coords, to_exprs = dummyfy(from_coords, to_exprs)
        self.transforms[to_sys] = (Matrix(from_coords), Matrix(to_exprs))
        if inverse:
            to_sys.transforms[self] = self._inv_transf(from_coords, to_exprs)
        if fill_in_gaps:
            self._fill_gaps_in_transformations()

    @staticmethod
    def _inv_transf(from_coords, to_exprs):
        inv_from = [i.as_dummy() for i in from_coords]
        inv_to = solve([t[0] - t[1] for t in zip(inv_from, to_exprs)], list(from_coords), dict=True)[0]
        inv_to = [inv_to[fc] for fc in from_coords]
        return (Matrix(inv_from), Matrix(inv_to))

    @staticmethod
    def _fill_gaps_in_transformations():
        raise NotImplementedError

    def transform(self, sys, coordinates=None):
        """
        Return the result of coordinate transformation from *self* to *sys*.
        If coordinates are not given, coordinate symbols of *self* are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.ImmutableDenseMatrix containing CoordinateSymbol

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.transform(R2_p)
        Matrix([
        [sqrt(x**2 + y**2)],
        [      atan2(y, x)]])
        >>> R2_r.transform(R2_p, [0, 1])
        Matrix([
        [   1],
        [pi/2]])

        """
        if coordinates is None:
            coordinates = self.symbols
        if self != sys:
            transf = self.transformation(sys)
            coordinates = transf(*coordinates)
        else:
            coordinates = Matrix(coordinates)
        return coordinates

    def coord_tuple_transform_to(self, to_sys, coords):
        """Transform ``coords`` to coord system ``to_sys``."""
        sympy_deprecation_warning('\n            The CoordSystem.coord_tuple_transform_to() method is deprecated.\n            Use the CoordSystem.transform() method instead.\n            ', deprecated_since_version='1.7', active_deprecations_target='deprecated-diffgeom-mutable')
        coords = Matrix(coords)
        if self != to_sys:
            with ignore_warnings(SymPyDeprecationWarning):
                transf = self.transforms[to_sys]
            coords = transf[1].subs(list(zip(transf[0], coords)))
        return coords

    def jacobian(self, sys, coordinates=None):
        """
        Return the jacobian matrix of a transformation on given coordinates.
        If coordinates are not given, coordinate symbols of *self* are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.ImmutableDenseMatrix

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_p.jacobian(R2_r)
        Matrix([
        [cos(theta), -rho*sin(theta)],
        [sin(theta),  rho*cos(theta)]])
        >>> R2_p.jacobian(R2_r, [1, 0])
        Matrix([
        [1, 0],
        [0, 1]])

        """
        result = self.transform(sys).jacobian(self.symbols)
        if coordinates is not None:
            result = result.subs(list(zip(self.symbols, coordinates)))
        return result
    jacobian_matrix = jacobian

    def jacobian_determinant(self, sys, coordinates=None):
        """
        Return the jacobian determinant of a transformation on given
        coordinates. If coordinates are not given, coordinate symbols of *self*
        are used.

        Parameters
        ==========

        sys : CoordSystem

        coordinates : Any iterable, optional.

        Returns
        =======

        sympy.Expr

        Examples
        ========

        >>> from sympy.diffgeom.rn import R2_r, R2_p
        >>> R2_r.jacobian_determinant(R2_p)
        1/sqrt(x**2 + y**2)
        >>> R2_r.jacobian_determinant(R2_p, [1, 0])
        1

        """
        return self.jacobian(sys, coordinates).det()

    def point(self, coords):
        """Create a ``Point`` with coordinates given in this coord system."""
        return Point(self, coords)

    def point_to_coords(self, point):
        """Calculate the coordinates of a point in this coord system."""
        return point.coords(self)

    def base_scalar(self, coord_index):
        """Return ``BaseScalarField`` that takes a point and returns one of the coordinates."""
        return BaseScalarField(self, coord_index)
    coord_function = base_scalar

    def base_scalars(self):
        """Returns a list of all coordinate functions.
        For more details see the ``base_scalar`` method of this class."""
        return [self.base_scalar(i) for i in range(self.dim)]
    coord_functions = base_scalars

    def base_vector(self, coord_index):
        """Return a basis vector field.
        The basis vector field for this coordinate system. It is also an
        operator on scalar fields."""
        return BaseVectorField(self, coord_index)

    def base_vectors(self):
        """Returns a list of all base vectors.
        For more details see the ``base_vector`` method of this class."""
        return [self.base_vector(i) for i in range(self.dim)]

    def base_oneform(self, coord_index):
        """Return a basis 1-form field.
        The basis one-form field for this coordinate system. It is also an
        operator on vector fields."""
        return Differential(self.coord_function(coord_index))

    def base_oneforms(self):
        """Returns a list of all base oneforms.
        For more details see the ``base_oneform`` method of this class."""
        return [self.base_oneform(i) for i in range(self.dim)]