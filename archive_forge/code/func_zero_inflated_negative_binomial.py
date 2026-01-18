import os
import numpy as np
def zero_inflated_negative_binomial():
    obj = Namespace()
    obj.params = [1.883859, -10.280888, -0.204769, 1.137985, 1.344457]
    obj.llf = -44077.91
    obj.bse = [0.3653, 1.6694, 0.02178, 0.01163, 0.0217496]
    obj.aic = 88165.81
    return obj