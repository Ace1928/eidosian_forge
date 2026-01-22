import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
def _initialize_crt_transfer_primatives(client, config):
    lock = acquire_crt_s3_process_lock(PROCESS_LOCK_NAME)
    if lock is None:
        return (None, None)
    session = Session()
    region_name = client.meta.region_name
    credentials = client._get_credentials()
    serializer = _create_crt_request_serializer(session, region_name)
    s3_client = _create_crt_s3_client(session, config, region_name, credentials, lock)
    return (serializer, s3_client)