from pyomo.common.dependencies import (
import platform
import pyomo.common.unittest as unittest
import sys
import os
import pyomo.contrib.parmest.parmest as parmest
import pyomo.contrib.parmest.graphics as graphics
def test_pairwise_plot(self):
    graphics.pairwise_plot(self.A, alpha=0.8, distributions=['Rect', 'MVN', 'KDE'])