import pickle
import sys
import os
import io
import subprocess
import json
from functools import lru_cache
from typing import Any
from itertools import groupby
import base64
import warnings
import {add_local_files} from "https://cdn.jsdelivr.net/gh/pytorch/pytorch@main/torch/utils/viz/MemoryViz.js"
def trace_plot(data, device=None, plot_segments=False):
    """Generate a visualization over time of the memory usage recorded by the trace as an html file.

    Args:
        data: Memory snapshot as generated from torch.cuda.memory._snapshot()
        device (torch.device, optional): Generate the trace for this device, needed if multiple devices have allocations.
        plot_segments (bool, optional): Plots memory returned from cudaMalloc, rather than individual allocations.
                                        Defaults to False.

    Returns:
        str: HTML of visualization
    """
    return _format_viz(data, 'Active Memory Timeline' if not plot_segments else 'Active Cached Memory Timeline', device)