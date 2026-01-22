import inspect
from _pydev_bundle import pydev_log
from _pydevd_bundle.pydevd_comm import CMD_SET_BREAK, CMD_ADD_EXCEPTION_BREAK
from _pydevd_bundle.pydevd_constants import STATE_SUSPEND, DJANGO_SUSPEND, \
from _pydevd_bundle.pydevd_frame_utils import add_exception_to_frame, FCode, just_raised, ignore_exception_trace
from pydevd_file_utils import canonical_normalized_path, absolute_path
from _pydevd_bundle.pydevd_api import PyDevdAPI
from pydevd_plugins.pydevd_line_validation import LineBreakpointWithLazyValidation, ValidationInfo
from _pydev_bundle.pydev_override import overrides
def _get_template_original_file_name_from_frame(frame):
    try:
        if IS_DJANGO19:
            if 'context' in frame.f_locals:
                context = frame.f_locals['context']
                if hasattr(context, '_has_included_template'):
                    back = frame.f_back
                    while back is not None and frame.f_code.co_name in ('render', '_render'):
                        locals = back.f_locals
                        if 'self' in locals:
                            self = locals['self']
                            if self.__class__.__name__ == 'Template' and hasattr(self, 'origin') and hasattr(self.origin, 'name'):
                                return _convert_to_str(self.origin.name)
                        back = back.f_back
                elif hasattr(context, 'template') and hasattr(context.template, 'origin') and hasattr(context.template.origin, 'name'):
                    return _convert_to_str(context.template.origin.name)
            return None
        elif IS_DJANGO19_OR_HIGHER:
            if 'self' in frame.f_locals:
                self = frame.f_locals['self']
                if hasattr(self, 'origin') and hasattr(self.origin, 'name'):
                    return _convert_to_str(self.origin.name)
            return None
        source = _get_source_django_18_or_lower(frame)
        if source is None:
            pydev_log.debug('Source is None\n')
            return None
        fname = _convert_to_str(source[0].name)
        if fname == '<unknown source>':
            pydev_log.debug('Source name is %s\n' % fname)
            return None
        else:
            return fname
    except:
        if DebugInfoHolder.DEBUG_TRACE_LEVEL >= 2:
            pydev_log.exception('Error getting django template filename.')
        return None