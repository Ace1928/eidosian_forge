import numpy as np
def update_interpolators(self):
    self.xi = Interpolator(self.tt, self.xx)
    self.yi = Interpolator(self.tt, self.yy)