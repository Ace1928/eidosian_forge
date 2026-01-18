from .plot_interval import PlotInterval
from .plot_object import PlotObject
from .util import parse_option_string
from sympy.core.symbol import Symbol
from sympy.core.sympify import sympify
from sympy.geometry.entity import GeometryEntity
from sympy.utilities.iterables import is_sequence

        Initializes the plot mode based on
        the 'mode-specific parameters' above.
        Only intended to be called by
        PlotMode._register(). To use a mode without
        registering it, you can directly call
        ModeSubclass._init_mode().
        