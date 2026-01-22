import contextlib
import functools
import pluggy
import keyring.errors

    >>> pluggy._hooks.varnames(devpiclient_get_password)
    (('url', 'username'), ())
    >>>
    