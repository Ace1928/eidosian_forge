import sys
import time
import warnings
from math import cos, sin, atan, tan, degrees, pi, sqrt
from typing import Dict, Any
import numpy as np
from ase.optimize.optimize import Optimizer
from ase.parallel import world
from ase.calculators.singlepoint import SinglePointCalculator
from ase.utils import IOContext
class MinModeControl(IOContext):
    """A parent class for controlling minimum mode saddle point searches.

    Method specific control classes inherit this one. The only thing
    inheriting classes need to implement are the log() method and
    the *parameters* class variable with default values for ALL
    parameters needed by the method in question.
    When instantiating control classes default parameter values can
    be overwritten.

    """
    parameters: Dict[str, Any] = {}

    def __init__(self, logfile='-', eigenmode_logfile=None, **kwargs):
        for key in kwargs:
            if not key in self.parameters:
                e = 'Invalid parameter >>%s<< with value >>%s<< in %s' % (key, str(kwargs[key]), self.__class__.__name__)
                raise ValueError(e)
            else:
                self.set_parameter(key, kwargs[key], log=False)
        self.initialize_logfiles(logfile, eigenmode_logfile)
        self.counters = {'forcecalls': 0, 'rotcount': 0, 'optcount': 0}
        self.log()

    def initialize_logfiles(self, logfile=None, eigenmode_logfile=None):
        self.logfile = self.openfile(logfile, comm=world)
        self.eigenmode_logfile = self.openfile(eigenmode_logfile, comm=world)

    def log(self, parameter=None):
        """Log the parameters of the eigenmode search."""
        pass

    def set_parameter(self, parameter, value, log=True):
        """Change a parameter's value."""
        if not parameter in self.parameters:
            e = 'Invalid parameter >>%s<< with value >>%s<<' % (parameter, str(value))
            raise ValueError(e)
        self.parameters[parameter] = value
        if log:
            self.log(parameter)

    def get_parameter(self, parameter):
        """Returns the value of a parameter."""
        if not parameter in self.parameters:
            e = 'Invalid parameter >>%s<<' % parameter
            raise ValueError(e)
        return self.parameters[parameter]

    def get_logfile(self):
        """Returns the log file."""
        return self.logfile

    def get_eigenmode_logfile(self):
        """Returns the eigenmode log file."""
        return self.eigenmode_logfile

    def get_counter(self, counter):
        """Returns a given counter."""
        return self.counters[counter]

    def increment_counter(self, counter):
        """Increment a given counter."""
        self.counters[counter] += 1

    def reset_counter(self, counter):
        """Reset a given counter."""
        self.counters[counter] = 0

    def reset_all_counters(self):
        """Reset all counters."""
        for key in self.counters:
            self.counters[key] = 0