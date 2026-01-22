from __future__ import annotations
import functools
import re
import typing as ty
import warnings
Return decorator function function for deprecation warning / error

        Parameters
        ----------
        message : str
            Message explaining deprecation, giving possible alternatives.
        since : str, optional
            Released version at which object was first deprecated.
        until : str, optional
            Last released version at which this function will still raise a
            deprecation warning.  Versions higher than this will raise an
            error.
        warn_class : None or class, optional
            Class of warning to generate for deprecation (overrides instance
            default).
        error_class : None or class, optional
            Class of error to generate when `version_comparator` returns 1 for a
            given argument of ``until`` (overrides class default).

        Returns
        -------
        deprecator : func
            Function returning a decorator.
        