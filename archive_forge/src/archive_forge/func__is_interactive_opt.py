from holoviews.core.overlay import CompositeOverlay
from holoviews.core.options import Store
from holoviews.plotting.util import COLOR_ALIASES
def _is_interactive_opt(bk_opt):
    """
    Heuristics to detect if a bokeh option is about interactivity, like
    'selection_alpha'.

    >>> is_interactive_opt('height')
    False
    >>> is_interactive_opt('annular_muted_alpha')
    True
    """
    interactive_flags = ['hover', 'muted', 'nonselection', 'selection']
    return any((part in interactive_flags for part in bk_opt.split('_')))