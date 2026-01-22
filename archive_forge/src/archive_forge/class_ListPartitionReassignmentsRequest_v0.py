from .api import Request, Response
from .types import (
class ListPartitionReassignmentsRequest_v0(Request):
    FLEXIBLE_VERSION = True
    API_KEY = 46
    API_VERSION = 0
    RESPONSE_TYPE = ListPartitionReassignmentsResponse_v0
    SCHEMA = Schema(('timeout_ms', Int32), ('topics', CompactArray(('name', CompactString('utf-8')), ('partition_index', CompactArray(Int32)), ('tags', TaggedFields))), ('tags', TaggedFields))