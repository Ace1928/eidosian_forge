import sys
from _pydev_bundle import pydev_log
def activate_pyplot():
    pyplot = sys.modules['matplotlib.pyplot']
    pyplot.show._needmain = False
    pyplot.draw_if_interactive = flag_calls(pyplot.draw_if_interactive)