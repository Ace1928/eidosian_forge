from __future__ import annotations
import typing
from abc import ABCMeta, abstractmethod
from ..exceptions import PlotnineError
def as_labeller(x: Optional[CanBeStripLabellingFunc]=None, default: CanBeStripLabellingFunc=label_value, multi_line: bool=True) -> labeller:
    """
    Coerse to labeller

    Parameters
    ----------
    x : callable | dict
        Object to coerce
    default : str | callable
        Default labeller. If it is a string,
        it should be the name of one the labelling
        functions provided by plotnine.
    multi_line : bool
        Whether to place each variable on a separate line

    Returns
    -------
    out : labeller
        Labelling function
    """
    if x is None:
        x = default
    if isinstance(x, labeller):
        return x
    x = _as_strip_labelling_func(x)
    return labeller(rows=x, cols=x, multi_line=multi_line)