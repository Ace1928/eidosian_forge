from __future__ import annotations
import typing
from .exceptions import PlotnineError
from .iapi import labels_view
from .mapping.aes import SCALED_AESTHETICS, rename_aesthetics
class xlab(labs):
    """
    Create x-axis label

    Parameters
    ----------
    xlab :
        x-axis label
    """

    def __init__(self, xlab: str):
        if xlab is None:
            raise PlotnineError('Arguments to xlab cannot be None')
        self.labels = labels_view(x=xlab)