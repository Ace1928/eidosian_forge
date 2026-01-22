import time
import numpy as np
import pyqtgraph as pg
from . import relax
class ChainSim(pg.QtCore.QObject):
    stepped = pg.QtCore.Signal()
    relaxed = pg.QtCore.Signal()

    def __init__(self):
        pg.QtCore.QObject.__init__(self)
        self.damping = 0.1
        self.relaxPerStep = 10
        self.maxTimeStep = 0.01
        self.pos = None
        self.mass = None
        self.fixed = None
        self.links = None
        self.lengths = None
        self.push = None
        self.pull = None
        self.initialized = False
        self.lasttime = None
        self.lastpos = None

    def init(self):
        if self.initialized:
            return
        if self.fixed is None:
            self.fixed = np.zeros(self.pos.shape[0], dtype=bool)
        if self.push is None:
            self.push = np.ones(self.links.shape[0], dtype=bool)
        if self.pull is None:
            self.pull = np.ones(self.links.shape[0], dtype=bool)
        l1 = self.links[:, 0]
        l2 = self.links[:, 1]
        m1 = self.mass[l1]
        m2 = self.mass[l2]
        self.mrel1 = (m1 / (m1 + m2))[:, np.newaxis]
        self.mrel1[self.fixed[l1]] = 1
        self.mrel1[self.fixed[l2]] = 0
        self.mrel2 = 1.0 - self.mrel1
        for i in range(10):
            self.relax(n=10)
        self.initialized = True

    def makeGraph(self):
        brushes = np.where(self.fixed, pg.mkBrush(0, 0, 0, 255), pg.mkBrush(50, 50, 200, 255))
        g2 = pg.GraphItem(pos=self.pos, adj=self.links[self.push & self.pull], pen=0.5, brush=brushes, symbol='o', size=self.mass ** 0.33, pxMode=False)
        p = pg.ItemGroup()
        p.addItem(g2)
        return p

    def update(self):
        now = time.perf_counter()
        if self.lasttime is None:
            dt = 0
        else:
            dt = now - self.lasttime
        self.lasttime = now
        if not relax.COMPILED:
            dt = self.maxTimeStep
        if self.lastpos is None:
            self.lastpos = self.pos
        fixedpos = self.pos[self.fixed]
        while dt > 0:
            dt1 = min(self.maxTimeStep, dt)
            dt -= dt1
            dx = self.pos - self.lastpos
            self.lastpos = self.pos
            acc = np.array([[0, -5]]) * dt1
            inertia = dx * (self.damping ** (dt1 / self.mass))[:, np.newaxis]
            self.pos = self.pos + inertia + acc
            self.pos[self.fixed] = fixedpos
            self.relax(self.relaxPerStep)
        self.stepped.emit()

    def relax(self, n=50):
        relax.relax(self.pos, self.links, self.mrel1, self.mrel2, self.lengths, self.push, self.pull, n)
        self.relaxed.emit()