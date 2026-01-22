import collections
import itertools
import numbers
import threading
import time
from typing import (
import warnings
import pandas as pd
from ortools.sat import cp_model_pb2
from ortools.sat import sat_parameters_pb2
from ortools.sat.python import cp_model_helper as cmh
from ortools.sat.python import swig_helper
from ortools.util.python import sorted_interval_list
class _ProductCst(LinearExpr):
    """Represents the product of a LinearExpr by a constant."""

    def __init__(self, expr, coeff):
        coeff = cmh.assert_is_a_number(coeff)
        if isinstance(expr, _ProductCst):
            self.__expr = expr.expression()
            self.__coef = expr.coefficient() * coeff
        else:
            self.__expr = expr
            self.__coef = coeff

    def __str__(self):
        if self.__coef == -1:
            return '-' + str(self.__expr)
        else:
            return '(' + str(self.__coef) + ' * ' + str(self.__expr) + ')'

    def __repr__(self):
        return f'ProductCst({self.__expr!r}, {self.__coef!r})'

    def coefficient(self):
        return self.__coef

    def expression(self):
        return self.__expr