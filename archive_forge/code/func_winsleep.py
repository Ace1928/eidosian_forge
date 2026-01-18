import re
import sys
import time
def winsleep():
    if sys.platform.startswith('win'):
        time.sleep(0.001)