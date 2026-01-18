from ..Qt import QtCore, QtWidgets
import weakref
def updateCanvasList(self):
    canvases = CanvasManager.instance().listCanvases()
    canvases.insert(0, '')
    if self.hostName in canvases:
        canvases.remove(self.hostName)
    sel = self.currentText()
    if sel in canvases:
        self.blockSignals(True)
    self.clear()
    for i in canvases:
        self.addItem(i)
        if i == sel:
            self.setCurrentIndex(self.count())
    self.blockSignals(False)