from .api import Request, Response
from .types import (
class CreateTopicsRequest_v1(Request):
    API_KEY = 19
    API_VERSION = 1
    RESPONSE_TYPE = CreateTopicsResponse_v1
    SCHEMA = Schema(('create_topic_requests', Array(('topic', String('utf-8')), ('num_partitions', Int32), ('replication_factor', Int16), ('replica_assignment', Array(('partition_id', Int32), ('replicas', Array(Int32)))), ('configs', Array(('config_key', String('utf-8')), ('config_value', String('utf-8')))))), ('timeout', Int32), ('validate_only', Boolean))