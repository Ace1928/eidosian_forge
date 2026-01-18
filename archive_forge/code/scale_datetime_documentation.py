from __future__ import annotations
from contextlib import suppress
from ..doctools import document
from .scale_continuous import scale_continuous

    Base class for all date/datetime scales

    Parameters
    ----------
    date_breaks : str, default=None
        A string giving the distance between major breaks.
        For example `'2 weeks'`, `'5 years'`. If specified,
        `date_breaks` takes precedence over
        `breaks`.
    date_labels : str, default=None
        Format string for the labels.
        See [strftime](:ref:`strftime-strptime-behavior`).
        If specified, `date_labels` takes precedence over
        `labels`.
    date_minor_breaks : str, default=None
        A string giving the distance between minor breaks.
        For example `'2 weeks'`, `'5 years'`. If specified,
        `date_minor_breaks` takes precedence over
        `minor_breaks`.
    {superclass_parameters}
    