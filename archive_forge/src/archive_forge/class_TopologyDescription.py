from __future__ import annotations
from random import sample
from typing import (
from bson.min_key import MinKey
from bson.objectid import ObjectId
from pymongo import common
from pymongo.errors import ConfigurationError
from pymongo.read_preferences import ReadPreference, _AggWritePref, _ServerMode
from pymongo.server_description import ServerDescription
from pymongo.server_selectors import Selection
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import _Address
class TopologyDescription:

    def __init__(self, topology_type: int, server_descriptions: dict[_Address, ServerDescription], replica_set_name: Optional[str], max_set_version: Optional[int], max_election_id: Optional[ObjectId], topology_settings: Any) -> None:
        """Representation of a deployment of MongoDB servers.

        :Parameters:
          - `topology_type`: initial type
          - `server_descriptions`: dict of (address, ServerDescription) for
            all seeds
          - `replica_set_name`: replica set name or None
          - `max_set_version`: greatest setVersion seen from a primary, or None
          - `max_election_id`: greatest electionId seen from a primary, or None
          - `topology_settings`: a TopologySettings
        """
        self._topology_type = topology_type
        self._replica_set_name = replica_set_name
        self._server_descriptions = server_descriptions
        self._max_set_version = max_set_version
        self._max_election_id = max_election_id
        self._topology_settings = topology_settings
        self._incompatible_err = None
        if self._topology_type != TOPOLOGY_TYPE.LoadBalanced:
            self._init_incompatible_err()
        readable_servers = self.readable_servers
        if not readable_servers:
            self._ls_timeout_minutes = None
        elif any((s.logical_session_timeout_minutes is None for s in readable_servers)):
            self._ls_timeout_minutes = None
        else:
            self._ls_timeout_minutes = min((s.logical_session_timeout_minutes for s in readable_servers))

    def _init_incompatible_err(self) -> None:
        """Internal compatibility check for non-load balanced topologies."""
        for s in self._server_descriptions.values():
            if not s.is_server_type_known:
                continue
            server_too_new = s.min_wire_version is not None and s.min_wire_version > common.MAX_SUPPORTED_WIRE_VERSION
            server_too_old = s.max_wire_version is not None and s.max_wire_version < common.MIN_SUPPORTED_WIRE_VERSION
            if server_too_new:
                self._incompatible_err = 'Server at %s:%d requires wire version %d, but this version of PyMongo only supports up to %d.' % (s.address[0], s.address[1] or 0, s.min_wire_version, common.MAX_SUPPORTED_WIRE_VERSION)
            elif server_too_old:
                self._incompatible_err = 'Server at %s:%d reports wire version %d, but this version of PyMongo requires at least %d (MongoDB %s).' % (s.address[0], s.address[1] or 0, s.max_wire_version, common.MIN_SUPPORTED_WIRE_VERSION, common.MIN_SUPPORTED_SERVER_VERSION)
                break

    def check_compatible(self) -> None:
        """Raise ConfigurationError if any server is incompatible.

        A server is incompatible if its wire protocol version range does not
        overlap with PyMongo's.
        """
        if self._incompatible_err:
            raise ConfigurationError(self._incompatible_err)

    def has_server(self, address: _Address) -> bool:
        return address in self._server_descriptions

    def reset_server(self, address: _Address) -> TopologyDescription:
        """A copy of this description, with one server marked Unknown."""
        unknown_sd = self._server_descriptions[address].to_unknown()
        return updated_topology_description(self, unknown_sd)

    def reset(self) -> TopologyDescription:
        """A copy of this description, with all servers marked Unknown."""
        if self._topology_type == TOPOLOGY_TYPE.ReplicaSetWithPrimary:
            topology_type = TOPOLOGY_TYPE.ReplicaSetNoPrimary
        else:
            topology_type = self._topology_type
        sds = {address: ServerDescription(address) for address in self._server_descriptions}
        return TopologyDescription(topology_type, sds, self._replica_set_name, self._max_set_version, self._max_election_id, self._topology_settings)

    def server_descriptions(self) -> dict[_Address, ServerDescription]:
        """dict of (address,
        :class:`~pymongo.server_description.ServerDescription`).
        """
        return self._server_descriptions.copy()

    @property
    def topology_type(self) -> int:
        """The type of this topology."""
        return self._topology_type

    @property
    def topology_type_name(self) -> str:
        """The topology type as a human readable string.

        .. versionadded:: 3.4
        """
        return TOPOLOGY_TYPE._fields[self._topology_type]

    @property
    def replica_set_name(self) -> Optional[str]:
        """The replica set name."""
        return self._replica_set_name

    @property
    def max_set_version(self) -> Optional[int]:
        """Greatest setVersion seen from a primary, or None."""
        return self._max_set_version

    @property
    def max_election_id(self) -> Optional[ObjectId]:
        """Greatest electionId seen from a primary, or None."""
        return self._max_election_id

    @property
    def logical_session_timeout_minutes(self) -> Optional[int]:
        """Minimum logical session timeout, or None."""
        return self._ls_timeout_minutes

    @property
    def known_servers(self) -> list[ServerDescription]:
        """List of Servers of types besides Unknown."""
        return [s for s in self._server_descriptions.values() if s.is_server_type_known]

    @property
    def has_known_servers(self) -> bool:
        """Whether there are any Servers of types besides Unknown."""
        return any((s for s in self._server_descriptions.values() if s.is_server_type_known))

    @property
    def readable_servers(self) -> list[ServerDescription]:
        """List of readable Servers."""
        return [s for s in self._server_descriptions.values() if s.is_readable]

    @property
    def common_wire_version(self) -> Optional[int]:
        """Minimum of all servers' max wire versions, or None."""
        servers = self.known_servers
        if servers:
            return min((s.max_wire_version for s in self.known_servers))
        return None

    @property
    def heartbeat_frequency(self) -> int:
        return self._topology_settings.heartbeat_frequency

    @property
    def srv_max_hosts(self) -> int:
        return self._topology_settings._srv_max_hosts

    def _apply_local_threshold(self, selection: Optional[Selection]) -> list[ServerDescription]:
        if not selection:
            return []
        fastest = min((cast(float, s.round_trip_time) for s in selection.server_descriptions))
        threshold = self._topology_settings.local_threshold_ms / 1000.0
        return [s for s in selection.server_descriptions if cast(float, s.round_trip_time) - fastest <= threshold]

    def apply_selector(self, selector: Any, address: Optional[_Address]=None, custom_selector: Optional[_ServerSelector]=None) -> list[ServerDescription]:
        """List of servers matching the provided selector(s).

        :Parameters:
          - `selector`: a callable that takes a Selection as input and returns
            a Selection as output. For example, an instance of a read
            preference from :mod:`~pymongo.read_preferences`.
          - `address` (optional): A server address to select.
          - `custom_selector` (optional): A callable that augments server
            selection rules. Accepts a list of
            :class:`~pymongo.server_description.ServerDescription` objects and
            return a list of server descriptions that should be considered
            suitable for the desired operation.

        .. versionadded:: 3.4
        """
        if getattr(selector, 'min_wire_version', 0):
            common_wv = self.common_wire_version
            if common_wv and common_wv < selector.min_wire_version:
                raise ConfigurationError("%s requires min wire version %d, but topology's min wire version is %d" % (selector, selector.min_wire_version, common_wv))
        if isinstance(selector, _AggWritePref):
            selector.selection_hook(self)
        if self.topology_type == TOPOLOGY_TYPE.Unknown:
            return []
        elif self.topology_type in (TOPOLOGY_TYPE.Single, TOPOLOGY_TYPE.LoadBalanced):
            return self.known_servers
        if address:
            description = self.server_descriptions().get(address)
            return [description] if description else []
        selection = Selection.from_topology_description(self)
        if self.topology_type != TOPOLOGY_TYPE.Sharded:
            selection = selector(selection)
        if custom_selector is not None and selection:
            selection = selection.with_server_descriptions(custom_selector(selection.server_descriptions))
        return self._apply_local_threshold(selection)

    def has_readable_server(self, read_preference: _ServerMode=ReadPreference.PRIMARY) -> bool:
        """Does this topology have any readable servers available matching the
        given read preference?

        :Parameters:
          - `read_preference`: an instance of a read preference from
            :mod:`~pymongo.read_preferences`. Defaults to
            :attr:`~pymongo.read_preferences.ReadPreference.PRIMARY`.

        .. note:: When connected directly to a single server this method
          always returns ``True``.

        .. versionadded:: 3.4
        """
        common.validate_read_preference('read_preference', read_preference)
        return any(self.apply_selector(read_preference))

    def has_writable_server(self) -> bool:
        """Does this topology have a writable server available?

        .. note:: When connected directly to a single server this method
          always returns ``True``.

        .. versionadded:: 3.4
        """
        return self.has_readable_server(ReadPreference.PRIMARY)

    def __repr__(self) -> str:
        servers = sorted(self._server_descriptions.values(), key=lambda sd: sd.address)
        return '<{} id: {}, topology_type: {}, servers: {!r}>'.format(self.__class__.__name__, self._topology_settings._topology_id, self.topology_type_name, servers)