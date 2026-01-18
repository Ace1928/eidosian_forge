from typing import cast, Optional, Sequence, SupportsFloat, Union
import collections
import numpy as np
import matplotlib.pyplot as plt
import cirq.study.result as result
Plot the state histogram from either a single result with repetitions or
       a histogram computed using `result.histogram()` or a flattened histogram
       of measurement results computed using `get_state_histogram`.

    Args:
        data:   The histogram values to plot. Possible options are:
                `result.Result`: Histogram is computed using
                    `get_state_histogram` and all 2 ** num_qubits values are
                    plotted, including 0s.
                `collections.Counter`: Only (key, value) pairs present in
                    collection are plotted.
                `Sequence[SupportsFloat]`: Values in the input sequence are
                    plotted. i'th entry corresponds to height of the i'th
                    bar in histogram.
        ax:      The Axes to plot on. If not given, a new figure is created,
                 plotted on, and shown.
        tick_label: Tick labels for the histogram plot in case input is not
                    `collections.Counter`. By default, label for i'th entry
                     is |i>.
        xlabel:  Label for the x-axis.
        ylabel:  Label for the y-axis.
        title:   Title of the plot.

    Returns:
        The axis that was plotted on.
    