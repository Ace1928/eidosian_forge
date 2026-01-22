from __future__ import annotations
import functools
import re
import typing as ty
import warnings
class Deprecator:
    """Class to make decorator marking function or method as deprecated

    The decorated function / method will:

    * Raise the given `warning_class` warning when the function / method gets
      called, up to (and including) version `until` (if specified);
    * Raise the given `error_class` error when the function / method gets
      called, when the package version is greater than version `until` (if
      specified).

    Parameters
    ----------
    version_comparator : callable
        Callable accepting string as argument, and return 1 if string
        represents a higher version than encoded in the `version_comparator`, 0
        if the version is equal, and -1 if the version is lower.  For example,
        the `version_comparator` may compare the input version string to the
        current package version string.
    warn_class : class, optional
        Class of warning to generate for deprecation.
    error_class : class, optional
        Class of error to generate when `version_comparator` returns 1 for a
        given argument of ``until`` in the ``__call__`` method (see below).
    """

    def __init__(self, version_comparator: ty.Callable[[str], int], warn_class: type[Warning]=DeprecationWarning, error_class: type[Exception]=ExpiredDeprecationError) -> None:
        self.version_comparator = version_comparator
        self.warn_class = warn_class
        self.error_class = error_class

    def is_bad_version(self, version_str: str) -> bool:
        """Return True if `version_str` is too high

        Tests `version_str` with ``self.version_comparator``

        Parameters
        ----------
        version_str : str
            String giving version to test

        Returns
        -------
        is_bad : bool
            True if `version_str` is for version below that expected by
            ``self.version_comparator``, False otherwise.
        """
        return self.version_comparator(version_str) == -1

    def __call__(self, message: str, since: str='', until: str='', warn_class: type[Warning] | None=None, error_class: type[Exception] | None=None) -> ty.Callable[[ty.Callable[P, T]], ty.Callable[P, T]]:
        """Return decorator function function for deprecation warning / error

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
        """
        exception = error_class if error_class is not None else self.error_class
        warning = warn_class if warn_class is not None else self.warn_class
        messages = [message]
        if (since, until) != ('', ''):
            messages.append('')
        if since:
            messages.append('* deprecated from version: ' + since)
        if until:
            messages.append(f'* {('Raises' if self.is_bad_version(until) else 'Will raise')} {exception} as of version: {until}')
        message = '\n'.join(messages)

        def deprecator(func: ty.Callable[P, T]) -> ty.Callable[P, T]:

            @functools.wraps(func)
            def deprecated_func(*args: P.args, **kwargs: P.kwargs) -> T:
                if until and self.is_bad_version(until):
                    raise exception(message)
                warnings.warn(message, warning, stacklevel=2)
                return func(*args, **kwargs)
            keep_doc = deprecated_func.__doc__
            if keep_doc is None:
                keep_doc = ''
            setup = TESTSETUP
            cleanup = TESTCLEANUP
            if keep_doc and until and self.is_bad_version(until):
                lines = '\n'.join((line.rstrip() for line in keep_doc.splitlines()))
                keep_doc = lines.split('\n\n', 1)[0]
                setup = ''
                cleanup = ''
            deprecated_func.__doc__ = _add_dep_doc(keep_doc, message, setup, cleanup)
            return deprecated_func
        return deprecator