from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, JINJA2_SUSPEND
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from pydevd_file_utils import canonical_normalized_path
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode
from _pydev_bundle import pydev_log
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
from _pydevd_bundle.pydevd_api import PyDevdAPI
def _find_render_function_frame(frame):
    old_frame = frame
    try:
        while not ('self' in frame.f_locals and frame.f_locals['self'].__class__.__name__ == 'Template' and (frame.f_code.co_name == 'render')):
            frame = frame.f_back
            if frame is None:
                return old_frame
        return frame
    except:
        return old_frame