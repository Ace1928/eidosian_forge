import threading
import pyomo.common.unittest as unittest
from pyomo.common.multithread import *
from threading import Thread
from pyomo.opt.base.solvers import check_available_solvers
def test_wrapper_field(self):
    sut = MultiThreadWrapper(Dummy)
    self.assertEqual(sut.number, 1)