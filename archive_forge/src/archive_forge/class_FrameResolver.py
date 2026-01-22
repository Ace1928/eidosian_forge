from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_utils import hasattr_checked, DAPGrouper, Timer
from io import StringIO
import traceback
from os.path import basename
from functools import partial
from _pydevd_bundle.pydevd_constants import IS_PY36_OR_GREATER, \
from _pydevd_bundle.pydevd_safe_repr import SafeRepr
from _pydevd_bundle import pydevd_constants
class FrameResolver:
    """
    This resolves a frame.
    """

    def resolve(self, obj, attribute):
        if attribute == '__internals__':
            return defaultResolver.get_dictionary(obj)
        if attribute == 'stack':
            return self.get_frame_stack(obj)
        if attribute == 'f_locals':
            return obj.f_locals
        return None

    def get_dictionary(self, obj):
        ret = {}
        ret['__internals__'] = defaultResolver.get_dictionary(obj)
        ret['stack'] = self.get_frame_stack(obj)
        ret['f_locals'] = obj.f_locals
        return ret

    def get_frame_stack(self, frame):
        ret = []
        if frame is not None:
            ret.append(self.get_frame_name(frame))
            while frame.f_back:
                frame = frame.f_back
                ret.append(self.get_frame_name(frame))
        return ret

    def get_frame_name(self, frame):
        if frame is None:
            return 'None'
        try:
            name = basename(frame.f_code.co_filename)
            return 'frame: %s [%s:%s]  id:%s' % (frame.f_code.co_name, name, frame.f_lineno, id(frame))
        except:
            return 'frame object'