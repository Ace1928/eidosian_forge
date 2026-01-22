import numpy as np
import pyqtgraph as pg
import pyqtgraph.metaarray as metaarray
from pyqtgraph.flowchart import Flowchart
from pyqtgraph.Qt import QtWidgets

This example demonstrates a very basic use of flowcharts: filter data,
displaying both the input and output of the filter. The behavior of
the filter can be reprogrammed by the user.

Basic steps are:
  - create a flowchart and two plots
  - input noisy data to the flowchart
  - flowchart connects data to the first plot, where it is displayed
  - add a gaussian filter to lowpass the data, then display it in the second plot.
