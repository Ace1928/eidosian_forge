import numpy as np
import pyqtgraph as pg

Display a plot and an image with minimal setup. 

pg.plot() and pg.image() are indended to be used from an interactive prompt
to allow easy data inspection (but note that PySide unfortunately does not
call the Qt event loop while the interactive prompt is running, in this case
it is necessary to call QApplication.exec_() to make the windows appear).
