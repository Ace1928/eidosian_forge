from .api import Request, Response
from .types import (
class AlterPartitionReassignmentsRequest_v0(Request):
    FLEXIBLE_VERSION = True
    API_KEY = 45
    API_VERSION = 0
    RESPONSE_TYPE = AlterPartitionReassignmentsResponse_v0
    SCHEMA = Schema(('timeout_ms', Int32), ('topics', CompactArray(('name', CompactString('utf-8')), ('partitions', CompactArray(('partition_index', Int32), ('replicas', CompactArray(Int32)), ('tags', TaggedFields))), ('tags', TaggedFields))), ('tags', TaggedFields))