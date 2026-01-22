from pyomo.contrib.pynumero.interfaces.utils import (
import numpy as np
import logging
import time
from pyomo.contrib.pynumero.linalg.base import LinearSolverStatus
from pyomo.common.timing import HierarchicalTimer
import enum
class LinearSolveContext(object):

    def __init__(self, interior_point_logger, linear_solver_logger, filename=None, level=logging.INFO):
        self.interior_point_logger = interior_point_logger
        self.linear_solver_logger = linear_solver_logger
        self.filename = filename
        if filename:
            self.handler = logging.FileHandler(filename)
            self.handler.setLevel(level)

    def __enter__(self):
        self.linear_solver_logger.propagate = False
        self.interior_point_logger.propagate = False
        if self.filename:
            self.linear_solver_logger.addHandler(self.handler)
            self.interior_point_logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        self.linear_solver_logger.propagate = True
        self.interior_point_logger.propagate = True
        if self.filename:
            self.linear_solver_logger.removeHandler(self.handler)
            self.interior_point_logger.removeHandler(self.handler)