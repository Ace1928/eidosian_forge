import threading
import pyomo.common.unittest as unittest
from pyomo.common.multithread import *
from threading import Thread
from pyomo.opt.base.solvers import check_available_solvers
def test_special_methods(self):
    sut = MultiThreadWrapper(Dummy)
    self.assertTrue(set(Dummy().__dir__()).issubset(set(sut.__dir__())))
    self.assertEqual(str(sut), str(Dummy()))