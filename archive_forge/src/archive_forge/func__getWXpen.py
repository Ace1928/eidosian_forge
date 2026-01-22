from wxPython.wx import *
from rdkit.sping import pid as sping_pid
def _getWXpen(self, width, color, default_color=None):
    """Converts PIDDLE colors to a wx pen"""
    if width is None or width < 0:
        width = self.defaultLineWidth
    if color == sping_pid.transparent:
        return wxTRANSPARENT_PEN
    wxcolor = self._getWXcolor(color)
    if wxcolor is None:
        if default_color is not None:
            return self._getWXpen(width, default_color)
        else:
            raise WxCanvasError('Cannot create pen.')
    return wxPen(wxcolor, width)