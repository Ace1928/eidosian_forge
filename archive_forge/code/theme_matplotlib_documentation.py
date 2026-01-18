from ..options import get_option
from .elements import element_blank, element_line, element_rect, element_text
from .theme import theme

    The default matplotlib look and feel.

    The theme can be used (and has the same parameter
    to customize) like a [](`matplotlib.rc_context`) manager.

    Parameters
    ----------
    rc : dict
        rcParams which should be applied on top of mathplotlib default.
    fname : str
        Filename to a matplotlibrc file
    use_defaults : bool
        If `True` (the default) resets the plot setting
        to the (current) `matplotlib.rcParams` values
    