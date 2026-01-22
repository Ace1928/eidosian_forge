from greenlet import greenlet
from . import TestCase
def Yield(value):
    g = greenlet.getcurrent()
    while not isinstance(g, genlet):
        if g is None:
            raise RuntimeError('yield outside a genlet')
        g = g.parent
    g.parent.switch(value)