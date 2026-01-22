import threading
import botocore.exceptions
from botocore.session import Session
from s3transfer.crt import (
class CRTS3Client:
    """
    This wrapper keeps track of our underlying CRT client, the lock used to
    acquire it and the region we've used to instantiate the client.

    Due to limitations in the existing CRT interfaces, we can only make calls
    in a single region and does not support redirects. We track the region to
    ensure we don't use the CRT client when a successful request cannot be made.
    """

    def __init__(self, crt_client, process_lock, region, cred_provider):
        self.crt_client = crt_client
        self.process_lock = process_lock
        self.region = region
        self.cred_provider = cred_provider