from functools import partial
import numpy as np
def _flag_blue(x):
    return 0.75 * np.sin((x * 31.5 - 0.25) * np.pi) + 0.5