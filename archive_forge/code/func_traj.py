import math
from plotly import exceptions, optional_imports
from plotly.figure_factory import utils
from plotly.graph_objs import graph_objs
def traj(self, xb, yb):
    """
        Integrate trajectories

        :param (int) xb: results of passing xi through self.blank_pos
        :param (int) xy: results of passing yi through self.blank_pos

        Calculate each trajectory based on rk4 integrate method.
        """
    if xb < 0 or xb >= self.density or yb < 0 or (yb >= self.density):
        return
    if self.blank[yb, xb] == 0:
        t = self.rk4_integrate(xb * self.spacing_x, yb * self.spacing_y)
        if t is not None:
            self.trajectories.append(t)