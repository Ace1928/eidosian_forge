import numpy as np
from scipy import stats
from scipy.special import comb
from scipy.stats.distributions import rv_continuous
import matplotlib.pyplot as plt
from numpy import where, inf
from numpy import abs as np_abs
def warnif(cond, msg):
    if not cond:
        print(msg, 'does not hold')