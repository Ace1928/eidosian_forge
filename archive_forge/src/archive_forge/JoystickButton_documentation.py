import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

JoystickButton is a button with x/y values. When the button is depressed and the
mouse dragged, the x/y values change to follow the mouse.
When the mouse button is released, the x/y values change to 0,0 (rather like 
letting go of the joystick).
