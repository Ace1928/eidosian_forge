from __future__ import annotations
from collections import abc
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from pymongo import max_staleness_selectors
from pymongo.errors import ConfigurationError
from pymongo.server_selectors import (
class ReadPreference:
    """An enum that defines some commonly used read preference modes.

    Apps can also create a custom read preference, for example::

       Nearest(tag_sets=[{"node":"analytics"}])

    See :doc:`/examples/high_availability` for code examples.

    A read preference is used in three cases:

    :class:`~pymongo.mongo_client.MongoClient` connected to a single mongod:

    - ``PRIMARY``: Queries are allowed if the server is standalone or a replica
      set primary.
    - All other modes allow queries to standalone servers, to a replica set
      primary, or to replica set secondaries.

    :class:`~pymongo.mongo_client.MongoClient` initialized with the
    ``replicaSet`` option:

    - ``PRIMARY``: Read from the primary. This is the default, and provides the
      strongest consistency. If no primary is available, raise
      :class:`~pymongo.errors.AutoReconnect`.

    - ``PRIMARY_PREFERRED``: Read from the primary if available, or if there is
      none, read from a secondary.

    - ``SECONDARY``: Read from a secondary. If no secondary is available,
      raise :class:`~pymongo.errors.AutoReconnect`.

    - ``SECONDARY_PREFERRED``: Read from a secondary if available, otherwise
      from the primary.

    - ``NEAREST``: Read from any member.

    :class:`~pymongo.mongo_client.MongoClient` connected to a mongos, with a
    sharded cluster of replica sets:

    - ``PRIMARY``: Read from the primary of the shard, or raise
      :class:`~pymongo.errors.OperationFailure` if there is none.
      This is the default.

    - ``PRIMARY_PREFERRED``: Read from the primary of the shard, or if there is
      none, read from a secondary of the shard.

    - ``SECONDARY``: Read from a secondary of the shard, or raise
      :class:`~pymongo.errors.OperationFailure` if there is none.

    - ``SECONDARY_PREFERRED``: Read from a secondary of the shard if available,
      otherwise from the shard primary.

    - ``NEAREST``: Read from any shard member.
    """
    PRIMARY = Primary()
    PRIMARY_PREFERRED = PrimaryPreferred()
    SECONDARY = Secondary()
    SECONDARY_PREFERRED = SecondaryPreferred()
    NEAREST = Nearest()