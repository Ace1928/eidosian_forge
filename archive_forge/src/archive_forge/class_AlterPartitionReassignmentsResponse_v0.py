from .api import Request, Response
from .types import (
class AlterPartitionReassignmentsResponse_v0(Response):
    API_KEY = 45
    API_VERSION = 0
    SCHEMA = Schema(('throttle_time_ms', Int32), ('error_code', Int16), ('error_message', CompactString('utf-8')), ('responses', CompactArray(('name', CompactString('utf-8')), ('partitions', CompactArray(('partition_index', Int32), ('error_code', Int16), ('error_message', CompactString('utf-8')), ('tags', TaggedFields))), ('tags', TaggedFields))), ('tags', TaggedFields))