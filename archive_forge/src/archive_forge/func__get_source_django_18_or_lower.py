import inspect
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, DJANGO_SUSPEND, \
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode, just_raised, ignore_exception_trace
from pydevd_file_utils import canonical_normalized_path, absolute_path
from _pydevd_bundle.pydevd_api import PyDevdAPI
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
def _get_source_django_18_or_lower(frame):
    try:
        node = frame.f_locals['self']
        if hasattr(node, 'source'):
            return node.source
        else:
            if IS_DJANGO18:
                pydev_log.error_once("WARNING: Template path is not available. Set the 'debug' option in the OPTIONS of a DjangoTemplates backend.")
            else:
                pydev_log.error_once('WARNING: Template path is not available. Please set TEMPLATE_DEBUG=True in your settings.py to make django template breakpoints working')
            return None
    except:
        pydev_log.exception()
        return None