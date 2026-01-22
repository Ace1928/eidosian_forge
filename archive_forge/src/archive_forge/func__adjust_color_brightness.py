import datetime
import matplotlib.dates  as mdates
import matplotlib.colors as mcolors
import numpy as np
def _adjust_color_brightness(color, amount=0.5):

    def _adjcb(c1, amount):
        import matplotlib.colors as mc
        import colorsys
        try:
            c = mc.cnames[c1]
        except:
            c = c1
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    if not isinstance(color, (list, tuple)):
        return _adjcb(color, amount)
    cout = []
    cadj = {}
    for c1 in color:
        if c1 in cadj:
            cout.append(cadj[c1])
        else:
            newc = _adjcb(c1, amount)
            cadj[c1] = newc
            cout.append(cadj[c1])
    return cout