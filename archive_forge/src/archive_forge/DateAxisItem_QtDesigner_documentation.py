import os
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets, loadUiType

Demonstrates the usage of DateAxisItem in a layout created with Qt Designer.

The spotlight here is on the 'setAxisItems' method, without which
one would have to subclass plotWidget in order to attach a dateaxis to it.
