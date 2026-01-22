from ._base import DirectivePlugin
from ..toc import normalize_toc_item, render_toc_ul
def _normalize_level(options, name, default):
    level = options.get(name)
    if not level:
        return default
    try:
        return int(level)
    except (ValueError, TypeError):
        raise ValueError(f'"{name}" option MUST be integer')