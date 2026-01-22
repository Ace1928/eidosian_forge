import json
import urllib.parse as urllib_parse
def _build_debug_options(flags):
    """Build string representation of debug options from the launch config."""
    return ';'.join((DEBUG_OPTIONS_BY_FLAG[flag] for flag in flags or [] if flag in DEBUG_OPTIONS_BY_FLAG))