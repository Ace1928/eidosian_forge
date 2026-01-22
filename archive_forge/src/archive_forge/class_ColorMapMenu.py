import importlib.util
import re
import numpy as np
from .. import colormap
from .. import functions as fn
from ..graphicsItems.GradientPresets import Gradients
from ..Qt import QtCore, QtGui, QtWidgets
class ColorMapMenu(QtWidgets.QMenu):

    def __init__(self, showGradientSubMenu=False):
        super().__init__()
        topmenu = self
        act = topmenu.addAction('None')
        act.setData((None, None))
        topmenu.addSeparator()
        if showGradientSubMenu:
            submenu = topmenu.addMenu('preset gradient')
            submenu.aboutToShow.connect(self.buildGradientSubMenu)
        submenu = topmenu.addMenu('local')
        submenu.aboutToShow.connect(self.buildLocalSubMenu)
        have_colorcet = importlib.util.find_spec('colorcet') is not None
        if not have_colorcet:
            submenu = topmenu.addMenu('cet (local)')
            submenu.aboutToShow.connect(self.buildCetLocalSubMenu)
        else:
            submenu = topmenu.addMenu('cet (external)')
            submenu.aboutToShow.connect(self.buildCetExternalSubMenu)
        if importlib.util.find_spec('matplotlib') is not None:
            submenu = topmenu.addMenu('matplotlib')
            submenu.aboutToShow.connect(self.buildMatplotlibSubMenu)
        if have_colorcet:
            submenu = topmenu.addMenu('colorcet')
            submenu.aboutToShow.connect(self.buildColorcetSubMenu)

    def buildGradientSubMenu(self):
        source = 'preset-gradient'
        names = list(Gradients.keys())
        self.buildSubMenu(names, source, sort=False)

    def buildLocalSubMenu(self):
        source = None
        names = colormap.listMaps(source=source)
        names = [x for x in names if not x.startswith('CET')]
        self.buildSubMenu(names, source)

    def buildCetLocalSubMenu(self):
        source = None
        names = colormap.listMaps(source=source)
        names = [x for x in names if x.startswith('CET')]
        self.buildSubMenu(names, source)

    def buildCetExternalSubMenu(self):
        source = 'colorcet'
        names = colormap.listMaps(source=source)
        names = [x for x in names if x.startswith('CET')]
        self.buildSubMenu(names, source)

    def buildMatplotlibSubMenu(self):
        source = 'matplotlib'
        names = colormap.listMaps(source=source)
        names = [x for x in names if not x.startswith('cet_')]
        names = [x for x in names if not x.endswith('_r')]
        self.buildSubMenu(names, source)

    def buildColorcetSubMenu(self):
        source = 'colorcet'
        import colorcet
        names = list(colorcet.palette_n.keys())
        self.buildSubMenu(names, source)

    def buildSubMenu(self, names, source, sort=True):
        menu = self.sender()
        menu.aboutToShow.disconnect()
        if sort:
            pattern = re.compile('(\\d+)')
            key = lambda x: [int(c) if c.isdigit() else c for c in pattern.split(x)]
            names = sorted(names, key=key)
        for name in names:
            if source == 'preset-gradient':
                cmap = preset_gradient_to_colormap(name)
            else:
                cmap = colormap.get(name, source=source)
            act = QtWidgets.QWidgetAction(menu)
            act.setData((name, source))
            act.setDefaultWidget(buildMenuEntryWidget(cmap, name))
            menu.addAction(act)