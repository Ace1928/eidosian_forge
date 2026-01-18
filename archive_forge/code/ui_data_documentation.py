import logging
from pyomo.common.collections import ComponentMap
from pyomo.contrib.viewer.qt import *
import pyomo.environ as pyo

            Start automatically emitting update signal again when properties
            are changed and emit update for changes made between begin_update
            and end_update
            