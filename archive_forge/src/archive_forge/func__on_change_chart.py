import json
import anywidget
import traitlets
import pathlib
from typing import Any, Set, Optional
import altair as alt
from altair.utils._vegafusion_data import (
from altair import TopLevelSpec
from altair.utils.selection import IndexSelection, PointSelection, IntervalSelection
@traitlets.observe('chart')
def _on_change_chart(self, change):
    """
        Internal callback function that updates the JupyterChart's internal
        state when the wrapped Chart instance changes
        """
    new_chart = change.new
    selection_watches = []
    selection_types = {}
    initial_params = {}
    initial_vl_selections = {}
    empty_selections = {}
    if new_chart is None:
        with self.hold_sync():
            self.spec = None
            self._selection_types = selection_types
            self._vl_selections = initial_vl_selections
            self._params = initial_params
        return
    params = getattr(new_chart, 'params', [])
    if params is not alt.Undefined:
        for param in new_chart.params:
            if isinstance(param.name, alt.ParameterName):
                clean_name = param.name.to_json().strip('"')
            else:
                clean_name = param.name
            select = getattr(param, 'select', alt.Undefined)
            if select != alt.Undefined:
                if not isinstance(select, dict):
                    select = select.to_dict()
                select_type = select['type']
                if select_type == 'point':
                    if not (select.get('fields', None) or select.get('encodings', None)):
                        selection_types[clean_name] = 'index'
                        empty_selections[clean_name] = IndexSelection(name=clean_name, value=[], store=[])
                    else:
                        selection_types[clean_name] = 'point'
                        empty_selections[clean_name] = PointSelection(name=clean_name, value=[], store=[])
                elif select_type == 'interval':
                    selection_types[clean_name] = 'interval'
                    empty_selections[clean_name] = IntervalSelection(name=clean_name, value={}, store=[])
                else:
                    raise ValueError(f'Unexpected selection type {select.type}')
                selection_watches.append(clean_name)
                initial_vl_selections[clean_name] = {'value': None, 'store': []}
            else:
                clean_value = param.value if param.value != alt.Undefined else None
                initial_params[clean_name] = clean_value
    for param_name in collect_transform_params(new_chart):
        initial_params[param_name] = None
    self.params = Params(initial_params)

    def on_param_traitlet_changed(param_change):
        new_params = dict(self._params)
        new_params[param_change['name']] = param_change['new']
        self._params = new_params
    self.params.observe(on_param_traitlet_changed)
    self.selections = Selections(empty_selections)
    with self.hold_sync():
        if using_vegafusion():
            if self.local_tz is None:
                self.spec = None

                def on_local_tz_change(change):
                    self._init_with_vegafusion(change['new'])
                self.observe(on_local_tz_change, ['local_tz'])
            else:
                self._init_with_vegafusion(self.local_tz)
        else:
            self.spec = new_chart.to_dict()
        self._selection_types = selection_types
        self._vl_selections = initial_vl_selections
        self._params = initial_params