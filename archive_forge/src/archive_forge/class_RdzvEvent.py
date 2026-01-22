import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, Union, Optional
@dataclass
class RdzvEvent:
    """
    Dataclass to represent any rendezvous event.

    Args:
        name: Event name. (E.g. Current action being performed)
        run_id: The run id of the rendezvous
        message: The message describing the event
        hostname: Hostname of the node
        pid: The process id of the node
        node_state: The state of the node (INIT, RUNNING, SUCCEEDED, FAILED)
        master_endpoint: The master endpoint for the rendezvous store, if known
        rank: The rank of the node, if known
        local_id: The local_id of the node, if defined in dynamic_rendezvous.py
        error_trace: Error stack trace, if this is an error event.
    """
    name: str
    run_id: str
    message: str
    hostname: str
    pid: int
    node_state: NodeState
    master_endpoint: str = ''
    rank: Optional[int] = None
    local_id: Optional[int] = None
    error_trace: str = ''

    def __str__(self):
        return self.serialize()

    @staticmethod
    def deserialize(data: Union[str, 'RdzvEvent']) -> 'RdzvEvent':
        if isinstance(data, RdzvEvent):
            return data
        if isinstance(data, str):
            data_dict = json.loads(data)
        data_dict['node_state'] = NodeState[data_dict['node_state']]
        return RdzvEvent(**data_dict)

    def serialize(self) -> str:
        return json.dumps(asdict(self))