import sys
from _pydev_bundle import pydev_log
def activate_pylab():
    pylab = sys.modules['pylab']
    pylab.show._needmain = False
    pylab.draw_if_interactive = flag_calls(pylab.draw_if_interactive)