import sys
import logging
import queue
import threading
import time
import grpc
from typing import TYPE_CHECKING
import ray.core.generated.ray_client_pb2 as ray_client_pb2
import ray.core.generated.ray_client_pb2_grpc as ray_client_pb2_grpc
from ray.util.debug import log_once
def _process_rpc_error(self, e: grpc.RpcError) -> bool:
    """
        Processes RPC errors that occur while reading from data stream.
        Returns True if the error can be recovered from, False otherwise.
        """
    if self.client_worker._can_reconnect(e):
        if log_once('lost_reconnect_logs'):
            logger.warning('Log channel is reconnecting. Logs produced while the connection was down can be found on the head node of the cluster in `ray_client_server_[port].out`')
        logger.debug('Log channel dropped, retrying.')
        time.sleep(0.5)
        return True
    logger.debug('Shutting down log channel.')
    if not self.client_worker._in_shutdown:
        logger.exception('Unexpected exception:')
    return False