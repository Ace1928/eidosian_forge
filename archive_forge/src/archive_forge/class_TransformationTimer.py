import functools
import logging
import sys
import time
import traceback
from pyomo.common.deprecation import deprecation_warning
from pyomo.common.modeling import NOTSET as _NotSpecified
class TransformationTimer(object):
    __slots__ = ('obj', 'mode', 'timer')
    msg = '%6.*f seconds to apply Transformation %s%s'
    in_progress = 'TransformationTimer object for %s%s; %0.3f elapsed seconds'

    def __init__(self, obj, mode=None):
        self.obj = obj
        if mode is None:
            self.mode = ''
        else:
            self.mode = ' (%s)' % (mode,)
        self.timer = -default_timer()

    def report(self):
        self.timer += default_timer()
        _transform_logger.info(self)

    @property
    def name(self):
        return self.obj.__class__.__name__

    def __str__(self):
        total_time = self.timer
        if total_time < 0:
            total_time += default_timer()
            return self.in_progress % (self.name, self.mode, total_time)
        return self.msg % (2 if total_time >= 0.005 else 0, total_time, self.name, self.mode)