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
def _update_rs_no_primary_from_member(sds: MutableMapping[_Address, ServerDescription], replica_set_name: Optional[str], server_description: ServerDescription) -> tuple[int, Optional[str]]:
    """RS without known primary. Update from a non-primary's response.

    Pass in a dict of ServerDescriptions, current replica set name, and the
    ServerDescription we are processing.

    Returns (new topology type, new replica_set_name).
    """
    topology_type = TOPOLOGY_TYPE.ReplicaSetNoPrimary
    if replica_set_name is None:
        replica_set_name = server_description.replica_set_name
    elif replica_set_name != server_description.replica_set_name:
        sds.pop(server_description.address)
        return (topology_type, replica_set_name)
    for address in server_description.all_hosts:
        if address not in sds:
            sds[address] = ServerDescription(address)
    if server_description.me and server_description.address != server_description.me:
        sds.pop(server_description.address)
    return (topology_type, replica_set_name)