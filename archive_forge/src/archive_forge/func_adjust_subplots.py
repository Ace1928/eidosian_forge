from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
def adjust_subplots(**kwds):
    import matplotlib.pyplot as plt
    passed_kwds = dict(bottom=0.05, top=0.925, left=0.05, right=0.95, hspace=0.2)
    passed_kwds.update(kwds)
    plt.subplots_adjust(**passed_kwds)