import numpy as np
def txy(self, tms=False):
    return zip((1000 if tms else 1) * self.tt, self.xx, self.yy)