from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI
class Jinja2TemplateSyntaxErrorFrame(object):
    IS_PLUGIN_FRAME = True

    def __init__(self, frame, exception_cls_name, filename, lineno, f_locals):
        self.f_code = FCode('Jinja2 %s' % (exception_cls_name,), filename)
        self.f_lineno = lineno
        self.f_back = frame
        self.f_globals = {}
        self.f_locals = f_locals
        self.f_trace = None