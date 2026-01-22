import pyglet.gl as pgl
from sympy.core import S
from sympy.plotting.pygletplot.plot_mode_base import PlotModeBase
def _on_calculate_cverts(self):
    if not self.verts or not self.color:
        return

    def set_work_len(n):
        self._calculating_cverts_len = float(n)

    def inc_work_pos():
        self._calculating_cverts_pos += 1.0
    set_work_len(1)
    self._calculating_cverts_pos = 0
    self.cverts = self.color.apply_to_surface(self.verts, self.u_set, self.v_set, set_len=set_work_len, inc_pos=inc_work_pos)
    self.push_solid(self.draw_verts(True, True))