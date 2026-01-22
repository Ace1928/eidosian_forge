from _pydevd_bundle.pydevd_breakpoints import LineBreakpoint
from _pydevd_bundle.pydevd_api import PyDevdAPI
import bisect
from _pydev_bundle import pydev_log
def _collect_valid_lines_in_template(self, template):
    lines_cache = getattr(template, '__pydevd_lines_cache__', None)
    if lines_cache is not None:
        lines, sorted_lines = lines_cache
        return (lines, sorted_lines)
    lines = self._collect_valid_lines_in_template_uncached(template)
    lines = frozenset(lines)
    sorted_lines = tuple(sorted(lines))
    template.__pydevd_lines_cache__ = (lines, sorted_lines)
    return (lines, sorted_lines)