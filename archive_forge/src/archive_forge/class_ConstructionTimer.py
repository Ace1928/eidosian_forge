import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
class ConstructionTimer(object):
    __slots__ = ('obj', 'timer')
    msg = '%6.*f seconds to construct %s %s%s'
    in_progress = 'ConstructionTimer object for %s %s; %0.3f elapsed seconds'

    def __init__(self, obj):
        self.obj = obj
        self.timer = -default_timer()

    def report(self):
        self.timer += default_timer()
        _construction_logger.info(self)

    @property
    def name(self):
        try:
            return self.obj.name
        except RuntimeError:
            try:
                return self.obj.local_name
            except RuntimeError:
                return '(unknown)'
        except AttributeError:
            return '(unknown)'

    def __str__(self):
        try:
            if self.obj.is_indexed():
                if self.obj.index_set().isfinite():
                    idx = len(self.obj.index_set())
                else:
                    idx = len(self.obj)
                idx_label = f'{idx} indices' if idx != 1 else '1 index'
            elif hasattr(self.obj, 'index_set'):
                idx = len(self.obj.index_set())
                idx_label = f'{idx} indices' if idx != 1 else '1 index'
            else:
                idx_label = ''
        except AttributeError:
            idx_label = ''
        if idx_label:
            idx_label = f'; {idx_label} total'
        try:
            _type = self.obj.ctype.__name__
        except AttributeError:
            _type = type(self.obj).__name__
        total_time = self.timer
        if total_time < 0:
            total_time += default_timer()
            return self.in_progress % (_type, self.name, total_time)
        return self.msg % (2 if total_time >= 0.005 else 0, total_time, _type, self.name, idx_label)