import sys
from collections import OrderedDict
from IPython.display import display
from ipywidgets import VBox
from ipywidgets import Image as ipyImage
from numpy import arange, issubdtype, array, column_stack, shape
from .figure import Figure
from .scales import Scale, LinearScale, Mercator
from .axes import Axis
from .marks import (Lines, Scatter, ScatterGL, Hist, Bars, OHLC, Pie, Map, Image,
from .toolbar import Toolbar
from .interacts import (BrushIntervalSelector, FastIntervalSelector,
from traitlets.utils.sentinel import Sentinel
import functools
def _create_selector(int_type, func, trait, **kwargs):
    """Create a selector of the specified type.

    Also attaches the function `func` as an `on_trait_change` listener
    for the trait `trait` of the selector.

    This is an internal function which should not be called by the user.

    Parameters
    ----------

    int_type: type
        The type of selector to be added.
    func: function
        The call back function. It should take at least two arguments. The name
        of the trait and the value of the trait are passed as arguments.
    trait: string
        The name of the Selector trait whose change triggers the
        call back function `func`.
    """
    interaction = _add_interaction(int_type, **kwargs)
    if func is not None:
        interaction.on_trait_change(func, trait)
    return interaction