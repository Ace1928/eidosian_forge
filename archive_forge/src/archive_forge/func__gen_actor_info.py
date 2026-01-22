import json
import logging
from collections import defaultdict
from typing import Set
from ray._private.protobuf_compat import message_to_dict
import ray
from ray._private.client_mode_hook import client_mode_hook
from ray._private.resource_spec import NODE_ID_PREFIX, HEAD_NODE_RESOURCE_NAME
from ray._private.utils import (
from ray._raylet import GlobalStateAccessor
from ray.core.generated import common_pb2
from ray.core.generated import gcs_pb2
from ray.util.annotations import DeveloperAPI
def _gen_actor_info(self, actor_table_data):
    """Parse actor table data.

        Returns:
            Information from actor table.
        """
    actor_info = {'ActorID': binary_to_hex(actor_table_data.actor_id), 'ActorClassName': actor_table_data.class_name, 'IsDetached': actor_table_data.is_detached, 'Name': actor_table_data.name, 'JobID': binary_to_hex(actor_table_data.job_id), 'Address': {'IPAddress': actor_table_data.address.ip_address, 'Port': actor_table_data.address.port, 'NodeID': binary_to_hex(actor_table_data.address.raylet_id)}, 'OwnerAddress': {'IPAddress': actor_table_data.owner_address.ip_address, 'Port': actor_table_data.owner_address.port, 'NodeID': binary_to_hex(actor_table_data.owner_address.raylet_id)}, 'State': gcs_pb2.ActorTableData.ActorState.DESCRIPTOR.values_by_number[actor_table_data.state].name, 'NumRestarts': actor_table_data.num_restarts, 'Timestamp': actor_table_data.timestamp, 'StartTime': actor_table_data.start_time, 'EndTime': actor_table_data.end_time, 'DeathCause': actor_table_data.death_cause, 'Pid': actor_table_data.pid}
    return actor_info