import copy
import re
import numpy as np
from plotly import colors
from ...core.util import isfinite, max_range
from ..util import color_intervals, process_cmap
def _compute_subplot_domains(widths, spacing):
    """
    Compute normalized domain tuples for a list of widths and a subplot
    spacing value

    Parameters
    ----------
    widths: list of float
        List of the desired widths of each subplot. The length of this list
        is also the specification of the number of desired subplots
    spacing: float
        Spacing between subplots in normalized coordinates

    Returns
    -------
    list of tuple of float
    """
    widths_sum = float(sum(widths))
    total_spacing = (len(widths) - 1) * spacing
    total_width = widths_sum + total_spacing
    relative_spacing = spacing / (widths_sum + total_spacing)
    relative_widths = [w / total_width for w in widths]
    domains = []
    for c in range(len(widths)):
        domain_start = c * relative_spacing + sum(relative_widths[:c])
        domain_stop = min(1, domain_start + relative_widths[c])
        domains.append((domain_start, domain_stop))
    return domains