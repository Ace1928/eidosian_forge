import csv
import itertools
import numpy as np
from .. import ErrorBarItem, PlotItem
from ..parametertree import Parameter
from ..Qt import QtCore
from .Exporter import Exporter
def _exportErrorBarItem(self, errorBarItem: ErrorBarItem) -> None:
    error_data = []
    index = next(self.index_counter)
    if errorBarItem.opts['x'] is None or errorBarItem.opts['y'] is None:
        return None
    header_naming_map = {'left': 'x_min_error', 'right': 'x_max_error', 'bottom': 'y_min_error', 'top': 'y_max_error'}
    self.header.extend([f'x{index:04}_error', f'y{index:04}_error'])
    error_data.extend([errorBarItem.opts['x'], errorBarItem.opts['y']])
    for error_direction, header_label in header_naming_map.items():
        if (error := errorBarItem.opts[error_direction]) is not None:
            self.header.extend([f'{header_label}_{index:04}'])
            error_data.append(error)
    self.data.append(tuple(error_data))
    return None