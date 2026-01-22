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
def _set_stage(self, index):
    if not index:
        return
    stage = self._progress_sel.source.iloc[index[0], 2]
    if stage in self.next_selector.options:
        self.next_selector.value = stage
        self.param.trigger('next')
    elif stage in self.prev_selector.options:
        self.prev_selector.value = stage
        self.param.trigger('previous')
    elif stage in self._route:
        while len(self._route) > 1:
            self.param.trigger('previous')
    else:
        route = find_route(self._graph, self._next_stage, stage)
        if route is None:
            route = find_route(self._graph, self._stage, stage)
            if route is None:
                raise ValueError('Could not find route to target node.')
        else:
            route = [self._next_stage] + route
        for r in route:
            if r not in self.next_selector.options:
                break
            self.next_selector.value = r
            self.param.trigger('next')