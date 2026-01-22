from .. import select
from .. import stats
from .. import utils
from .tools import label_axis
from .utils import _get_figure
from .utils import parse_fontsize
from .utils import shift_ticklabels
from .utils import show
from .utils import temp_fontsize
from scipy.cluster import hierarchy
import numpy as np
import pandas as pd
def _make_scatter_arrays(data_clust, cluster_names, tissues, markers, gene_names, normalize_emd, normalize_expression):
    cluster_labels = []
    marker_labels = []
    tissue_labels = []
    x = []
    y = []
    c = []
    s = []
    for j, marker in enumerate(markers):
        s_row = []
        c_row = []
        for i, cluster in enumerate(cluster_names):
            in_cluster_expr, out_cluster_expr = data_clust[cluster]
            x.append(i)
            y.append(j)
            marker_labels.append(marker)
            cluster_labels.append(cluster)
            if tissues is not None:
                tissue_labels.append(tissues[j])
            gidx = np.where(gene_names == marker)
            marker_expr = in_cluster_expr[:, gidx]
            s_row.append(stats.EMD(marker_expr, out_cluster_expr[:, gidx]))
            c_row.append(np.mean(marker_expr))
        s_row = np.array(s_row)
        if normalize_emd and np.max(s_row) != 0:
            s_row = 150 * s_row / np.max(s_row)
        c_row = np.array(c_row)
        if normalize_expression and np.max(c_row) != 0:
            c_row = c_row / np.max(c_row)
        s.append(s_row)
        c.append(c_row)
    s = np.concatenate(s)
    if not normalize_emd:
        s = 150 * s / np.max(s)
    c = np.concatenate(c)
    return (x, y, c, s, cluster_labels, tissue_labels, marker_labels)