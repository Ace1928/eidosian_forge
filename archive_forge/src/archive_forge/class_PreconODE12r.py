from ase.optimize.precon.precon import (Precon, Exp, C1, Pfrommer,
from ase.optimize.precon.lbfgs import PreconLBFGS
from ase.optimize.precon.fire import PreconFIRE
from ase.optimize.ode import ODE12r
class PreconODE12r(ODE12r):
    """
    Subclass of ase.optimize.ode.ODE12r with 'Exp' preconditioning on by default
    """

    def __init__(self, *args, **kwargs):
        if 'precon' not in kwargs:
            kwargs['precon'] = 'Exp'
        ODE12r.__init__(self, *args, **kwargs)