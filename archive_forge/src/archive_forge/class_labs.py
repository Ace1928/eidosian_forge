from __future__ import annotations
import typing
from .exceptions import PlotnineError
from .iapi import labels_view
from .mapping.aes import SCALED_AESTHETICS, rename_aesthetics
class labs:
    """
    Add labels for aesthetics and/or title

    Parameters
    ----------
    kwargs :
        Aesthetics (with scales) to be renamed. You can also
        set the `title` and `caption`.
    """
    labels: labels_view

    def __init__(self, **kwargs: str):
        unknown = kwargs.keys() - VALID_LABELS
        if unknown:
            raise PlotnineError(f'Cannot deal with these labels: {unknown}')
        self.labels = labels_view(**rename_aesthetics(kwargs))

    def __radd__(self, plot: p9.ggplot) -> p9.ggplot:
        """
        Add labels to ggplot object
        """
        plot.labels.update(self.labels)
        return plot