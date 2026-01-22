from greenlet import greenlet
from . import TestCase
from .leakcheck import fails_leakcheck
class TheGenlet(genlet):
    fn = (func,)