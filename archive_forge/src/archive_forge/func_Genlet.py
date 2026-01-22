from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
def Genlet(func):

    class TheGenlet(genlet):
        fn = (func,)
    return TheGenlet