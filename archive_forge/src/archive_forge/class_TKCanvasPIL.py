import tkFont
import Tkinter
import rdkit.sping.pid
class TKCanvasPIL(rdkit.sping.PIL.PILCanvas):
    """This canvas maintains a PILCanvas as its backbuffer.  Drawing calls
        are made to the backbuffer and flush() sends the image to the screen
        using TKCanvas.
           You can also save what is displayed to a file in any of the formats
        supported by PIL"""

    def __init__(self, size=(300, 300), name='TKCanvas', master=None, **kw):
        rdkit.sping.PIL.PILCanvas.__init__(self, size=size, name=name)
        self._tkcanvas = TKCanvas(size, name, master, **kw)

    def flush(self):
        rdkit.sping.PIL.PILCanvas.flush(self)
        self._tkcanvas.drawImage(self._image, 0, 0)
        self._tkcanvas.flush()

    def getTKCanvas(self):
        return self._tkcanvas