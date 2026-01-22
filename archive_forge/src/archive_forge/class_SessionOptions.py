from the same snapshot timestamp. The server chooses the latest
from __future__ import annotations
import collections
import time
import uuid
from collections.abc import Mapping as _Mapping
from typing import (
from bson.binary import Binary
from bson.int64 import Int64
from bson.son import SON
from bson.timestamp import Timestamp
from pymongo import _csot
from pymongo.cursor import _ConnectionManager
from pymongo.errors import (
from pymongo.helpers import _RETRYABLE_ERROR_CODES
from pymongo.read_concern import ReadConcern
from pymongo.read_preferences import ReadPreference, _ServerMode
from pymongo.server_type import SERVER_TYPE
from pymongo.write_concern import WriteConcern
class SessionOptions:
    """Options for a new :class:`ClientSession`.

    :Parameters:
      - `causal_consistency` (optional): If True, read operations are causally
        ordered within the session. Defaults to True when the ``snapshot``
        option is ``False``.
      - `default_transaction_options` (optional): The default
        TransactionOptions to use for transactions started on this session.
      - `snapshot` (optional): If True, then all reads performed using this
        session will read from the same snapshot. This option is incompatible
        with ``causal_consistency=True``. Defaults to ``False``.

    .. versionchanged:: 3.12
       Added the ``snapshot`` parameter.
    """

    def __init__(self, causal_consistency: Optional[bool]=None, default_transaction_options: Optional[TransactionOptions]=None, snapshot: Optional[bool]=False) -> None:
        if snapshot:
            if causal_consistency:
                raise ConfigurationError('snapshot reads do not support causal_consistency=True')
            causal_consistency = False
        elif causal_consistency is None:
            causal_consistency = True
        self._causal_consistency = causal_consistency
        if default_transaction_options is not None:
            if not isinstance(default_transaction_options, TransactionOptions):
                raise TypeError('default_transaction_options must be an instance of pymongo.client_session.TransactionOptions, not: {!r}'.format(default_transaction_options))
        self._default_transaction_options = default_transaction_options
        self._snapshot = snapshot

    @property
    def causal_consistency(self) -> bool:
        """Whether causal consistency is configured."""
        return self._causal_consistency

    @property
    def default_transaction_options(self) -> Optional[TransactionOptions]:
        """The default TransactionOptions to use for transactions started on
        this session.

        .. versionadded:: 3.7
        """
        return self._default_transaction_options

    @property
    def snapshot(self) -> Optional[bool]:
        """Whether snapshot reads are configured.

        .. versionadded:: 3.12
        """
        return self._snapshot