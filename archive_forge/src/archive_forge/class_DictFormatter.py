import numpy as np
from matplotlib import ticker as mticker
from matplotlib.transforms import Bbox, Transform
class DictFormatter:

    def __init__(self, format_dict, formatter=None):
        """
        format_dict : dictionary for format strings to be used.
        formatter : fall-back formatter
        """
        super().__init__()
        self._format_dict = format_dict
        self._fallback_formatter = formatter

    def __call__(self, direction, factor, values):
        """
        factor is ignored if value is found in the dictionary
        """
        if self._fallback_formatter:
            fallback_strings = self._fallback_formatter(direction, factor, values)
        else:
            fallback_strings = [''] * len(values)
        return [self._format_dict.get(k, v) for k, v in zip(values, fallback_strings)]