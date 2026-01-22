import copy
from dataclasses import astuple, dataclass
from typing import (
import matplotlib as mpl
import matplotlib.collections as mpl_collections
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import axes_grid1
from cirq.devices import grid_qubit
from cirq.vis import vis_utils
def _plot_on_axis(self, ax: plt.Axes) -> mpl_collections.Collection:
    polygon_list = self._get_polygon_units()
    collection: mpl_collections.Collection = mpl_collections.PolyCollection([c.polygon for c in polygon_list], **self._config.get('collection_options', {}))
    collection.set_clim(self._config.get('vmin'), self._config.get('vmax'))
    collection.set_array(np.array([c.value for c in polygon_list]))
    ax.add_collection(collection)
    collection.update_scalarmappable()
    if self._config.get('annotation_map') or self._config.get('annotation_format'):
        self._write_annotations([(c.center, c.annot) for c in polygon_list], collection, ax)
    ax.set(xlabel='column', ylabel='row')
    if self._config.get('plot_colorbar'):
        self._plot_colorbar(collection, ax)
    rows = set([q.row for qubits in self._value_map.keys() for q in qubits])
    cols = set([q.col for qubits in self._value_map.keys() for q in qubits])
    min_row, max_row = (min(rows), max(rows))
    min_col, max_col = (min(cols), max(cols))
    min_xtick = np.floor(min_col)
    max_xtick = np.ceil(max_col)
    ax.set_xticks(np.arange(min_xtick, max_xtick + 1))
    min_ytick = np.floor(min_row)
    max_ytick = np.ceil(max_row)
    ax.set_yticks(np.arange(min_ytick, max_ytick + 1))
    ax.set_xlim((min_xtick - 0.6, max_xtick + 0.6))
    ax.set_ylim((max_ytick + 0.6, min_ytick - 0.6))
    if self._config.get('title'):
        ax.set_title(self._config['title'], fontweight='bold')
    return collection