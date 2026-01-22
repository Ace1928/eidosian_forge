from wxPython.wx import *
from rdkit.sping import pid as sping_pid
def _setWXfont(self, font=None):
    """set/return the current font for the dc
        jjk  10/28/99"""
    wx_font = self._getWXfont(font)
    self.dc.SetFont(wx_font)
    return wx_font