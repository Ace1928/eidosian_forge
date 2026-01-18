from __future__ import (absolute_import, division, print_function)
import numpy as np
from ..util import import_
import pytest
from .. import ODESys
from ..core import integrate_chained
from ..symbolic import SymbolicSys, PartiallySolvedSystem, symmetricsys
from ..util import requires, pycvodes_double
from ._robertson import run_integration, get_ode_exprs
@requires('sym', 'sympy', 'pyodeint')
def test_run_integration():
    xout, yout, info = run_integration(integrator='odeint')[:3]
    assert info['success'] is True