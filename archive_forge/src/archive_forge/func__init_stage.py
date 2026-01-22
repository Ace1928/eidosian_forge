from __future__ import annotations
import sys
import traceback as tb
from collections import defaultdict
from typing import ClassVar, Tuple
import param
from .layout import Column, Row
from .pane import HoloViews, Markdown
from .param import Param
from .util import param_reprs
from .viewable import Viewer
from .widgets import Button, Select
def _init_stage(self):
    stage, stage_kwargs = self._stages[self._stage]
    previous = []
    for src, tgts in self._graph.items():
        if self._stage in tgts:
            previous.append(src)
    prev_states = [self._states[prev] for prev in previous if prev in self._states]
    outputs = []
    kwargs, results = ({}, {})
    for state in prev_states:
        for name, (_, method, index) in state.param.outputs().items():
            if name not in stage.param:
                continue
            if method not in results:
                results[method] = method()
            result = results[method]
            if index is not None:
                result = result[index]
            kwargs[name] = result
            outputs.append(name)
        if stage_kwargs.get('inherit_params', self.inherit_params):
            ignored = [stage_kwargs.get(p) or getattr(self, p, None) for p in ('ready_parameter', 'next_parameter')]
            params = [k for k, v in state.param.objects('existing').items() if k not in ignored]
            kwargs.update({k: v for k, v in state.param.values().items() if k in stage.param and k != 'name' and (k in params)})
    if isinstance(stage, param.Parameterized):
        stage.param.update(**kwargs)
        self._state = stage
    else:
        self._state = stage(**kwargs)
    for output in outputs:
        self._state.param[output].precedence = -1
    ready_param = stage_kwargs.get('ready_parameter', self.ready_parameter)
    if ready_param and ready_param in stage.param:
        self._state.param.watch(self._unblock, ready_param, onlychanged=False)
    next_param = stage_kwargs.get('next_parameter', self.next_parameter)
    if next_param and next_param in stage.param:
        self._state.param.watch(self._select_next, next_param, onlychanged=False)
    self._states[self._stage] = self._state
    return self._state.panel()