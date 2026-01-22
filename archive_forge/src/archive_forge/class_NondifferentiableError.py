import inspect
import textwrap
class NondifferentiableError(PyomoException, ValueError):
    """A Pyomo-specific ValueError raised for non-differentiable expressions"""
    pass