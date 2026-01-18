import sys
import datetime
from collections import namedtuple
def wl_increment(self):
    if self.wl_count < sys.maxint:
        self.wl_count += 1
    if self.wl_entered is None:
        self.wl_entered = datetime.datetime.now()
    self.wl_update()