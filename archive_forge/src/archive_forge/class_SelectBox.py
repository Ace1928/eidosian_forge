import importlib
from .. import ItemGroup, SRTTransform, debug
from .. import functions as fn
from ..graphicsItems.ROI import ROI
from ..Qt import QT_LIB, QtCore, QtWidgets
from . import TransformGuiTemplate_generic as ui_template
class SelectBox(ROI):

    def __init__(self, scalable=False, rotatable=True):
        ROI.__init__(self, [0, 0], [1, 1], invertible=True)
        center = [0.5, 0.5]
        if scalable:
            self.addScaleHandle([1, 1], center, lockAspect=True)
            self.addScaleHandle([0, 0], center, lockAspect=True)
        if rotatable:
            self.addRotateHandle([0, 1], center)
            self.addRotateHandle([1, 0], center)