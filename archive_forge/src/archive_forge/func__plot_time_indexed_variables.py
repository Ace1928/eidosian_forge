import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.contrib.incidence_analysis import IncidenceGraphInterface
from pyomo.common.dependencies.matplotlib import pyplot as plt
def _plot_time_indexed_variables(data, keys, show=False, save=False, fname=None, transparent=False):
    fig, ax = plt.subplots()
    time = data.get_time_points()
    for i, key in enumerate(keys):
        data_list = data.get_data_from_key(key)
        label = str(data.get_cuid(key))
        ax.plot(time, data_list, label=label)
    ax.legend()
    if show:
        plt.show()
    if save:
        if fname is None:
            fname = 'states.png'
        fig.savefig(fname, transparent=transparent)
    return (fig, ax)