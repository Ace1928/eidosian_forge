import abc
import logging
from pyomo.environ import SolverFactory
class _IISBase(abc.ABC):

    def __init__(self, solver):
        self._solver = solver

    @abc.abstractmethod
    def compute(self):
        """computes the IIS/Conflict"""
        pass

    @abc.abstractmethod
    def write(self, file_name):
        """writes the IIS in LP format
        return the file name written
        """
        pass