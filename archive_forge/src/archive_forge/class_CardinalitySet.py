import abc
import math
import functools
from numbers import Integral
from collections.abc import Iterable, MutableSequence
from enum import Enum
from pyomo.common.dependencies import numpy as np, scipy as sp
from pyomo.core.base import ConcreteModel, Objective, maximize, minimize, Block
from pyomo.core.base.constraint import ConstraintList
from pyomo.core.base.var import Var, IndexedVar
from pyomo.core.expr.numvalue import value, native_numeric_types
from pyomo.opt.results import check_optimal_termination
from pyomo.contrib.pyros.util import add_bounds_for_uncertain_parameters
class CardinalitySet(UncertaintySet):
    """
    A cardinality-constrained (a.k.a. "gamma") set.

    Parameters
    ----------
    origin : (N,) array_like
        Origin of the set (e.g., nominal uncertain parameter values).
    positive_deviation : (N,) array_like
        Maximal non-negative coordinate deviation from the origin
        in each dimension.
    gamma : numeric type
        Upper bound for the number of uncertain parameters which
        may realize their maximal deviations from the origin
        simultaneously.

    Examples
    --------
    A 3D cardinality set:

    >>> from pyomo.contrib.pyros import CardinalitySet
    >>> gamma_set = CardinalitySet(
    ...     origin=[0, 0, 0],
    ...     positive_deviation=[1.0, 2.0, 1.5],
    ...     gamma=1,
    ... )
    >>> gamma_set.origin
    array([0, 0, 0])
    >>> gamma_set.positive_deviation
    array([1. , 2. , 1.5])
    >>> gamma_set.gamma
    1
    """

    def __init__(self, origin, positive_deviation, gamma):
        """Initialize self (see class docstring)."""
        self.origin = origin
        self.positive_deviation = positive_deviation
        self.gamma = gamma

    @property
    def type(self):
        """
        str : Brief description of the type of the uncertainty set.
        """
        return 'cardinality'

    @property
    def origin(self):
        """
        (N,) numpy.ndarray : Origin of the cardinality set
        (e.g. nominal parameter values).
        """
        return self._origin

    @origin.setter
    def origin(self, val):
        validate_array(arr=val, arr_name='origin', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type')
        val_arr = np.array(val)
        if hasattr(self, '_origin'):
            if val_arr.size != self.dim:
                raise ValueError(f"Attempting to set attribute 'origin' of cardinality set of dimension {self.dim} to value of dimension {val_arr.size}")
        self._origin = val_arr

    @property
    def positive_deviation(self):
        """
        (N,) numpy.ndarray : Maximal coordinate deviations from the
        origin in each dimension. All entries are nonnegative.
        """
        return self._positive_deviation

    @positive_deviation.setter
    def positive_deviation(self, val):
        validate_array(arr=val, arr_name='positive_deviation', dim=1, valid_types=valid_num_types, valid_type_desc='a valid numeric type')
        for dev_val in val:
            if dev_val < 0:
                raise ValueError(f"Entry {dev_val} of attribute 'positive_deviation' is negative value")
        val_arr = np.array(val)
        if hasattr(self, '_origin'):
            if val_arr.size != self.dim:
                raise ValueError(f"Attempting to set attribute 'positive_deviation' of cardinality set of dimension {self.dim} to value of dimension {val_arr.size}")
        self._positive_deviation = val_arr

    @property
    def gamma(self):
        """
        numeric type : Upper bound for the number of uncertain
        parameters which may maximally deviate from their respective
        origin values simultaneously. Must be a numerical value ranging
        from 0 to the set dimension `N`.

        Note that, mathematically, setting `gamma` to 0 reduces the set
        to a singleton containing the center, while setting `gamma` to
        the set dimension `N` makes the set mathematically equivalent
        to a `BoxSet` with bounds
        ``numpy.array([origin, origin + positive_deviation]).T``.
        """
        return self._gamma

    @gamma.setter
    def gamma(self, val):
        validate_arg_type('gamma', val, valid_num_types, 'a valid numeric type', False)
        if val < 0 or val > self.dim:
            raise ValueError(f"Cardinality set attribute 'gamma' must be a real number between 0 and dimension {self.dim} (provided value {val})")
        self._gamma = val

    @property
    def dim(self):
        """
        int : Dimension `N` of the cardinality set.
        """
        return len(self.origin)

    @property
    def geometry(self):
        """
        Geometry of the cardinality set.
        See the `Geometry` class documentation.
        """
        return Geometry.LINEAR

    @property
    def parameter_bounds(self):
        """
        Bounds in each dimension of the cardinality set.

        Returns
        -------
        : list of tuples
            List, length `N`, of 2-tuples. Each tuple
            specifies the bounds in its corresponding
            dimension.
        """
        nom_val = self.origin
        deviation = self.positive_deviation
        gamma = self.gamma
        parameter_bounds = [(nom_val[i], nom_val[i] + min(gamma, 1) * deviation[i]) for i in range(len(nom_val))]
        return parameter_bounds

    def set_as_constraint(self, uncertain_params, **kwargs):
        """
        Construct a list of cardinality set constraints on
        a sequence of uncertain parameter objects.

        Parameters
        ----------
        uncertain_params : list of Param or list of Var
            Uncertain parameter objects upon which the constraints
            are imposed.
        **kwargs : dict
            Additional arguments. This dictionary should consist
            of a `model` entry, which maps to a `ConcreteModel`
            object representing the model of interest (parent model
            of the uncertain parameter objects).

        Returns
        -------
        conlist : ConstraintList
            The constraints on the uncertain parameters.
        """
        if len(uncertain_params) != len(self.origin):
            raise AttributeError('Dimensions of origin and uncertain_param lists must be equal.')
        model = kwargs['model']
        set_i = list(range(len(uncertain_params)))
        model.util.cassi = Var(set_i, initialize=0, bounds=(0, 1))
        conlist = ConstraintList()
        conlist.construct()
        for i in set_i:
            conlist.add(self.origin[i] + self.positive_deviation[i] * model.util.cassi[i] == uncertain_params[i])
        conlist.add(sum((model.util.cassi[i] for i in set_i)) <= self.gamma)
        return conlist

    def point_in_set(self, point):
        """
        Determine whether a given point lies in the cardinality set.

        Parameters
        ----------
        point : (N,) array-like
            Point (parameter value) of interest.

        Returns
        -------
        : bool
            True if the point lies in the set, False otherwise.
        """
        cassis = []
        for i in range(self.dim):
            if self.positive_deviation[i] > 0:
                cassis.append((point[i] - self.origin[i]) / self.positive_deviation[i])
        if sum((cassi for cassi in cassis)) <= self.gamma and all((cassi >= 0 and cassi <= 1 for cassi in cassis)):
            return True
        else:
            return False