from collections import OrderedDict
import functools
import numpy as np
from qiskit.utils import optionals as _optionals
from qiskit.result import QuasiDistribution, ProbDistribution
from .exceptions import VisualizationError
from .utils import matplotlib_close_if_inline
@_optionals.HAS_MATPLOTLIB.require_in_call
def _plotting_core(data, figsize=(7, 5), color=None, number_to_keep=None, sort='asc', target_string=None, legend=None, bar_labels=True, title=None, ax=None, filename=None, kind='counts'):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    if sort not in VALID_SORTS:
        raise VisualizationError("Value of sort option, %s, isn't a valid choice. Must be 'asc', 'desc', 'hamming', 'value', 'value_desc'")
    if sort in DIST_MEAS and target_string is None:
        err_msg = 'Must define target_string when using distance measure.'
        raise VisualizationError(err_msg)
    if isinstance(data, dict):
        data = [data]
    if legend and len(legend) != len(data):
        raise VisualizationError(f"Length of legend ({len(legend)}) doesn't match number of input executions ({len(data)}).")
    if color is None:
        color = ['#648fff', '#dc267f', '#785ef0', '#ffb000', '#fe6100']
    elif isinstance(color, str):
        color = [color]
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = None
    labels = sorted(functools.reduce(lambda x, y: x.union(y.keys()), data, set()))
    if number_to_keep is not None:
        labels.append('rest')
    if sort in DIST_MEAS:
        dist = []
        for item in labels:
            dist.append(DIST_MEAS[sort](item, target_string) if item != 'rest' else 0)
        labels = [list(x) for x in zip(*sorted(zip(dist, labels), key=lambda pair: pair[0]))][1]
    elif 'value' in sort:
        combined_counts = {}
        if isinstance(data, dict):
            combined_counts = data
        else:
            for counts in data:
                for count in counts:
                    prev_count = combined_counts.get(count, 0)
                    combined_counts[count] = max(prev_count, counts[count])
        labels = sorted(combined_counts.keys(), key=lambda key: combined_counts[key])
    length = len(data)
    width = 1 / (len(data) + 1)
    labels_dict, all_pvalues, all_inds = _plot_data(data, labels, number_to_keep, kind=kind)
    rects = []
    for item, _ in enumerate(data):
        label = None
        for idx, val in enumerate(all_pvalues[item]):
            if not idx and legend:
                label = legend[item]
            if val > 0:
                rects.append(ax.bar(idx + item * width, val, width, label=label, color=color[item % len(color)], zorder=2))
                label = None
        bar_center = width / 2 * (length - 1)
        ax.set_xticks(all_inds[item] + bar_center)
        ax.set_xticklabels(labels_dict.keys(), fontsize=14, rotation=70)
        if bar_labels:
            for rect in rects:
                for rec in rect:
                    height = rec.get_height()
                    if kind == 'distribution':
                        height = round(height, 3)
                    if height >= 0.001:
                        ax.text(rec.get_x() + rec.get_width() / 2.0, 1.05 * height, str(height), ha='center', va='bottom', zorder=3)
                    else:
                        ax.text(rec.get_x() + rec.get_width() / 2.0, 1.05 * height, '0', ha='center', va='bottom', zorder=3)
    if kind == 'counts':
        ax.set_ylabel('Count', fontsize=14)
    else:
        ax.set_ylabel('Quasi-probability', fontsize=14)
    all_vals = np.concatenate(all_pvalues).ravel()
    min_ylim = 0.0
    if kind == 'distribution':
        min_ylim = min(0.0, min((1.1 * val for val in all_vals)))
    ax.set_ylim([min_ylim, min([1.1 * sum(all_vals), max((1.1 * val for val in all_vals))])])
    if 'desc' in sort:
        ax.invert_xaxis()
    ax.yaxis.set_major_locator(MaxNLocator(5))
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(14)
    plt.grid(which='major', axis='y', zorder=0, linestyle='--')
    if title:
        plt.title(title)
    if legend:
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), ncol=1, borderaxespad=0, frameon=True, fontsize=12)
    if fig:
        matplotlib_close_if_inline(fig)
    if filename is None:
        return fig
    else:
        return fig.savefig(filename)