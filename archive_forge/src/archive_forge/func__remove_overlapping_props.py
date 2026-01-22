import ipywidgets as widgets
from traitlets import List, Unicode, Dict, observe, Integer
from .basedatatypes import BaseFigure, BasePlotlyType
from .callbacks import BoxSelector, LassoSelector, InputDeviceState, Points
from .serializers import custom_serializers
from .version import __frontend_version__
@staticmethod
def _remove_overlapping_props(input_data, delta_data, prop_path=()):
    """
        Remove properties in input_data that are also in delta_data, and do so
        recursively.

        Exception: Never remove 'uid' from input_data, this property is used
        to align traces

        Parameters
        ----------
        input_data : dict|list
        delta_data : dict|list

        Returns
        -------
        list[tuple[str|int]]
            List of removed property path tuples
        """
    removed = []
    if isinstance(input_data, dict):
        assert isinstance(delta_data, dict)
        for p, delta_val in delta_data.items():
            if isinstance(delta_val, dict) or BaseFigure._is_dict_list(delta_val):
                if p in input_data:
                    input_val = input_data[p]
                    recur_prop_path = prop_path + (p,)
                    recur_removed = BaseFigureWidget._remove_overlapping_props(input_val, delta_val, recur_prop_path)
                    removed.extend(recur_removed)
                    if not input_val:
                        input_data.pop(p)
                        removed.append(recur_prop_path)
            elif p in input_data and p != 'uid':
                input_data.pop(p)
                removed.append(prop_path + (p,))
    elif isinstance(input_data, list):
        assert isinstance(delta_data, list)
        for i, delta_val in enumerate(delta_data):
            if i >= len(input_data):
                break
            input_val = input_data[i]
            if input_val is not None and isinstance(delta_val, dict) or BaseFigure._is_dict_list(delta_val):
                recur_prop_path = prop_path + (i,)
                recur_removed = BaseFigureWidget._remove_overlapping_props(input_val, delta_val, recur_prop_path)
                removed.extend(recur_removed)
    return removed