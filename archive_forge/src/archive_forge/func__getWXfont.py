from wxPython.wx import *
from rdkit.sping import pid as sping_pid
def _getWXfont(self, font):
    """Returns a wxFont roughly equivalent to the requested PIDDLE font"""
    if font is None:
        font = self.defaultFont
    if font.face is None or font.face == 'times':
        family = wxDEFAULT
    elif font.face == 'courier' or font.face == 'monospaced':
        family = wxMODERN
    elif font.face == 'helvetica' or font.face == 'sansserif':
        family = wxSWISS
    elif font.face == 'serif' or font.face == 'symbol':
        family = wxDEFAULT
    else:
        family = wxDEFAULT
    weight = wxNORMAL
    style = wxNORMAL
    underline = 0
    if font.bold == 1:
        weight = wxBOLD
    if font.underline == 1:
        underline = 1
    if font.italic == 1:
        style = wxITALIC
    return wxFont(font.size, family, style, weight, underline)