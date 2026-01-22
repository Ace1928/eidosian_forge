from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI
class Jinja2LineBreakpoint(LineBreakpointWithLazyValidation):

    def __init__(self, canonical_normalized_filename, breakpoint_id, line, condition, func_name, expression, hit_condition=None, is_logpoint=False):
        self.canonical_normalized_filename = canonical_normalized_filename
        LineBreakpointWithLazyValidation.__init__(self, breakpoint_id, line, condition, func_name, expression, hit_condition=hit_condition, is_logpoint=is_logpoint)

    def __str__(self):
        return 'Jinja2LineBreakpoint: %s-%d' % (self.canonical_normalized_filename, self.line)