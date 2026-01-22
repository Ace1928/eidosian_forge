import inspect
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, DJANGO_SUSPEND, \
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode, just_raised, ignore_exception_trace
from pydevd_file_utils import canonical_normalized_path, absolute_path
from _pydevd_bundle.pydevd_api import PyDevdAPI
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
class DjangoTemplateFrame(object):
    IS_PLUGIN_FRAME = True

    def __init__(self, frame):
        original_filename = _get_template_original_file_name_from_frame(frame)
        self._back_context = frame.f_locals['context']
        self.f_code = FCode('Django Template', original_filename)
        self.f_lineno = _get_template_line(frame)
        self.f_back = frame
        self.f_globals = {}
        self.f_locals = self._collect_context(self._back_context)
        self.f_trace = None

    def _collect_context(self, context):
        res = {}
        try:
            for d in context.dicts:
                for k, v in d.items():
                    res[k] = v
        except AttributeError:
            pass
        return res

    def _change_variable(self, name, value):
        for d in self._back_context.dicts:
            for k, v in d.items():
                if k == name:
                    d[k] = value