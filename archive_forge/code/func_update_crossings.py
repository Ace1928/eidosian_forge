import os, time, webbrowser
from .gui import *
from . import smooth
from .vertex import Vertex
from .arrow import Arrow
from .crossings import Crossing, ECrossing
from .colors import Palette
from .dialog import InfoDialog
from .manager import LinkManager
from .viewer import LinkViewer
from .version import version
from .ipython_tools import IPythonTkRoot
def update_crossings(self, this_arrow):
    """
        Redraw any arrows which were changed by moving this_arrow.
        """
    if this_arrow == None:
        return
    cross_list = [c for c in self.Crossings if this_arrow in c]
    damage_list = []
    find = lambda x: cross_list[cross_list.index(x)]
    for arrow in self.Arrows:
        if this_arrow == arrow:
            continue
        new_crossing = Crossing(this_arrow, arrow)
        new_crossing.locate()
        if new_crossing.x != None:
            if new_crossing in cross_list:
                find(new_crossing).locate()
                continue
            else:
                self.Crossings.append(new_crossing)
        elif new_crossing in self.Crossings:
            if arrow == find(new_crossing).under:
                damage_list.append(arrow)
            self.Crossings.remove(new_crossing)
    for arrow in damage_list:
        arrow.draw(self.Crossings)