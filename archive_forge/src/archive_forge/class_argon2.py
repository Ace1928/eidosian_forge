from __future__ import with_statement, absolute_import
import logging
import re
import types
from warnings import warn
from passlib import exc
from passlib.crypto.digest import MAX_UINT32
from passlib.utils import classproperty, to_bytes, render_bytes
from passlib.utils.binary import b64s_encode, b64s_decode
from passlib.utils.compat import u, unicode, bascii_to_str, uascii_to_str, PY2
import passlib.utils.handlers as uh
class argon2(_NoBackend, _Argon2Common):
    """
    This class implements the Argon2 password hash [#argon2-home]_, and follows the :ref:`password-hash-api`.

    Argon2 supports a variable-length salt, and variable time & memory cost,
    and a number of other configurable parameters.

    The :meth:`~passlib.ifc.PasswordHash.replace` method accepts the following optional keywords:

    :type type: str
    :param type:
        Specify the type of argon2 hash to generate.
        Can be one of "ID", "I", "D".

        This defaults to "ID" if supported by the backend, otherwise "I".

    :type salt: str
    :param salt:
        Optional salt string.
        If specified, the length must be between 0-1024 bytes.
        If not specified, one will be auto-generated (this is recommended).

    :type salt_size: int
    :param salt_size:
        Optional number of bytes to use when autogenerating new salts.

    :type rounds: int
    :param rounds:
        Optional number of rounds to use.
        This corresponds linearly to the amount of time hashing will take.

    :type time_cost: int
    :param time_cost:
        An alias for **rounds**, for compatibility with underlying argon2 library.

    :param int memory_cost:
        Defines the memory usage in kibibytes.
        This corresponds linearly to the amount of memory hashing will take.

    :param int parallelism:
        Defines the parallelization factor.
        *NOTE: this will affect the resulting hash value.*

    :param int digest_size:
        Length of the digest in bytes.

    :param int max_threads:
        Maximum number of threads that will be used.
        -1 means unlimited; otherwise hashing will use ``min(parallelism, max_threads)`` threads.

        .. note::

            This option is currently only honored by the argon2pure backend.

    :type relaxed: bool
    :param relaxed:
        By default, providing an invalid value for one of the other
        keywords will result in a :exc:`ValueError`. If ``relaxed=True``,
        and the error can be corrected, a :exc:`~passlib.exc.PasslibHashWarning`
        will be issued instead. Correctable errors include ``rounds``
        that are too small or too large, and ``salt`` strings that are too long.

    .. versionchanged:: 1.7.2

        Added the "type" keyword, and support for type "D" and "ID" hashes.
        (Prior versions could verify type "D" hashes, but not generate them).

    .. todo::

        * Support configurable threading limits.
    """
    backends = ('argon2_cffi', 'argon2pure')
    _backend_mixin_target = True
    _backend_mixin_map = {None: _NoBackend, 'argon2_cffi': _CffiBackend, 'argon2pure': _PureBackend}