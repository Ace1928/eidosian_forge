from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from _pydevd_bundle.pydevd_api import PyDevdAPI
import bisect
from _pydev_bundle import pydev_log
def _verify_breakpoints_with_lines_collected(self, py_db, canonical_normalized_filename, template_breakpoints_for_file, valid_lines_frozenset, sorted_lines):
    for line, template_bp in list(template_breakpoints_for_file.items()):
        if template_bp.verified_cache_key != valid_lines_frozenset:
            template_bp.verified_cache_key = valid_lines_frozenset
            valid = line in valid_lines_frozenset
            if not valid:
                new_line = -1
                if sorted_lines:
                    idx = bisect.bisect_left(sorted_lines, line)
                    if idx > 0:
                        new_line = sorted_lines[idx - 1]
                if new_line >= 0 and new_line not in template_breakpoints_for_file:
                    if template_bp.add_breakpoint_result.error_code != PyDevdAPI.ADD_BREAKPOINT_NO_ERROR and template_bp.add_breakpoint_result.translated_line != new_line:
                        pydev_log.debug('Template breakpoint in %s in line: %s moved to line: %s', canonical_normalized_filename, line, new_line)
                        template_bp.add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_NO_ERROR
                        template_bp.add_breakpoint_result.translated_line = new_line
                        template_breakpoints_for_file.pop(line, None)
                        template_breakpoints_for_file[new_line] = template_bp
                        template_bp.on_changed_breakpoint_state(template_bp.breakpoint_id, template_bp.add_breakpoint_result)
                elif template_bp.add_breakpoint_result.error_code != PyDevdAPI.ADD_BREAKPOINT_INVALID_LINE:
                    pydev_log.debug('Template breakpoint in %s in line: %s invalid (valid lines: %s)', canonical_normalized_filename, line, valid_lines_frozenset)
                    template_bp.add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_INVALID_LINE
                    template_bp.on_changed_breakpoint_state(template_bp.breakpoint_id, template_bp.add_breakpoint_result)
            elif template_bp.add_breakpoint_result.error_code != PyDevdAPI.ADD_BREAKPOINT_NO_ERROR:
                template_bp.add_breakpoint_result.error_code = PyDevdAPI.ADD_BREAKPOINT_NO_ERROR
                template_bp.on_changed_breakpoint_state(template_bp.breakpoint_id, template_bp.add_breakpoint_result)