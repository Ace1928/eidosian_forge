from typing import Dict, List, Union, Optional, Any
from typing_extensions import Literal
from pydantic import BaseModel, Field
from rpcq.messages import TargetDevice as TargetQuantumProcessor
def _edge_ids_from_id(edge_id: str) -> List[int]:
    return [int(node_id) for node_id in edge_id.split('-')]