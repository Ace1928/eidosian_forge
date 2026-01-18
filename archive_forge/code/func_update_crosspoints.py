import time
from string import ascii_lowercase
from .gui import tkMessageBox
from .vertex import Vertex
from .arrow import Arrow, default_arrow_params
from .crossings import Crossing, ECrossing
from .smooth import TikZPicture
def update_crosspoints(self):
    for arrow in self.Arrows:
        arrow.vectorize()
        arrow.params = self.arrow_params
    for c in self.Crossings:
        c.locate()
    self.Crossings = [c for c in self.Crossings if c.x is not None]
    self.CrossPoints = [Vertex(c.x, c.y, self.canvas, style='hidden') for c in self.Crossings]