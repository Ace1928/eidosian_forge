from ..Qt import QtCore, QtWidgets
import weakref
class CanvasManager(QtCore.QObject):
    SINGLETON = None
    sigCanvasListChanged = QtCore.Signal()

    def __init__(self):
        if CanvasManager.SINGLETON is not None:
            raise Exception('Can only create one canvas manager.')
        QtCore.QObject.__init__(self)
        CanvasManager.SINGLETON = self
        self.canvases = weakref.WeakValueDictionary()

    @classmethod
    def instance(cls):
        return CanvasManager.SINGLETON

    def registerCanvas(self, canvas, name):
        n2 = name
        i = 0
        while n2 in self.canvases:
            n2 = '%s_%03d' % (name, i)
            i += 1
        self.canvases[n2] = canvas
        self.sigCanvasListChanged.emit()
        return n2

    def unregisterCanvas(self, name):
        c = self.canvases[name]
        del self.canvases[name]
        self.sigCanvasListChanged.emit()

    def listCanvases(self):
        return list(self.canvases.keys())

    def getCanvas(self, name):
        return self.canvases[name]