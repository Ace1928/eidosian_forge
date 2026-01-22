from collections import OrderedDict
import dateutil.parser
import itertools
import logging
import numpy as np
from matplotlib import _api, ticker, units
class StrCategoryLocator(ticker.Locator):
    """Tick at every integer mapping of the string data."""

    def __init__(self, units_mapping):
        """
        Parameters
        ----------
        units_mapping : dict
            Mapping of category names (str) to indices (int).
        """
        self._units = units_mapping

    def __call__(self):
        return list(self._units.values())

    def tick_values(self, vmin, vmax):
        return self()