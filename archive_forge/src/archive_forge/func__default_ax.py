import numpy as np
import shapely
def _default_ax():
    import matplotlib.pyplot as plt
    ax = plt.gca()
    ax.grid(True)
    ax.set_aspect('equal')
    return ax