from statsmodels.compat.python import lrange
import numpy as np
import statsmodels.tsa.vector_ar.util as util
class MPLConfigurator:

    def __init__(self):
        self._inverse_actions = []

    def revert(self):
        for action in self._inverse_actions:
            action()

    def set_fontsize(self, size):
        import matplotlib as mpl
        old_size = mpl.rcParams['font.size']
        mpl.rcParams['font.size'] = size

        def revert():
            mpl.rcParams['font.size'] = old_size
        self._inverse_actions.append(revert)