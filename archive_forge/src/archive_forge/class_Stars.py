import numpy as np
from matplotlib import _api
from matplotlib.path import Path
class Stars(Shapes):
    size = 1.0 / 3.0
    filled = True

    def __init__(self, hatch, density):
        self.num_rows = hatch.count('*') * density
        path = Path.unit_regular_star(5)
        self.shape_vertices = path.vertices
        self.shape_codes = np.full(len(self.shape_vertices), Path.LINETO, dtype=Path.code_type)
        self.shape_codes[0] = Path.MOVETO
        super().__init__(hatch, density)