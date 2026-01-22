from _pydev_bundle import pydev_log
from _pydevd_bundle import pydevd_import_class
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame
from _pydev_bundle._pydev_saved_modules import threading
class LineBreakpoint(object):

    def __init__(self, breakpoint_id, line, condition, func_name, expression, suspend_policy='NONE', hit_condition=None, is_logpoint=False):
        self.breakpoint_id = breakpoint_id
        self.line = line
        self.condition = condition
        self.func_name = func_name
        self.expression = expression
        self.suspend_policy = suspend_policy
        self.hit_condition = hit_condition
        self._hit_count = 0
        self._hit_condition_lock = threading.Lock()
        self.is_logpoint = is_logpoint

    @property
    def has_condition(self):
        return bool(self.condition) or bool(self.hit_condition)

    def handle_hit_condition(self, frame):
        if not self.hit_condition:
            return False
        ret = False
        with self._hit_condition_lock:
            self._hit_count += 1
            expr = self.hit_condition.replace('@HIT@', str(self._hit_count))
            try:
                ret = bool(eval(expr, frame.f_globals, frame.f_locals))
            except Exception:
                ret = False
        return ret