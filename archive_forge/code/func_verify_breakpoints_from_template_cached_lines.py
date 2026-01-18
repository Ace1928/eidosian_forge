from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from _pydevd_bundle.pydevd_api import PyDevdAPI
import bisect
from _pydev_bundle import pydev_log
def verify_breakpoints_from_template_cached_lines(self, py_db, canonical_normalized_filename, template_breakpoints_for_file):
    """
        This is used when the lines are already available (if just the template is available,
        `verify_breakpoints` should be used instead).
        """
    cached = self._canonical_normalized_filename_to_last_template_lines.get(canonical_normalized_filename)
    if cached is not None:
        valid_lines_frozenset, sorted_lines = cached
        self._verify_breakpoints_with_lines_collected(py_db, canonical_normalized_filename, template_breakpoints_for_file, valid_lines_frozenset, sorted_lines)