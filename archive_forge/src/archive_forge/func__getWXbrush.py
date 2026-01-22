from wxPython.wx import *
from rdkit.sping import pid as sping_pid
def _getWXbrush(self, color, default_color=None):
    """Converts PIDDLE colors to a wx brush"""
    if color == sping_pid.transparent:
        return wxTRANSPARENT_BRUSH
    wxcolor = self._getWXcolor(color)
    if wxcolor is None:
        if default_color is not None:
            return self._getWXbrush(default_color)
        else:
            raise WxCanvasError('Cannot create brush.')
    return wxBrush(wxcolor)