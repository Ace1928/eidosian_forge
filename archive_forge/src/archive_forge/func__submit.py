import copy
import math
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import calculate_range_parameter
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import ChunksizeAdjuster
def _submit(self, client, config, osutil, request_executor, transfer_future):
    """
        :param client: The client associated with the transfer manager

        :type config: s3transfer.manager.TransferConfig
        :param config: The transfer config associated with the transfer
            manager

        :type osutil: s3transfer.utils.OSUtil
        :param osutil: The os utility associated to the transfer manager

        :type request_executor: s3transfer.futures.BoundedExecutor
        :param request_executor: The request executor associated with the
            transfer manager

        :type transfer_future: s3transfer.futures.TransferFuture
        :param transfer_future: The transfer future associated with the
            transfer request that tasks are being submitted for
        """
    if transfer_future.meta.size is None:
        call_args = transfer_future.meta.call_args
        head_object_request = self._get_head_object_request_from_copy_source(call_args.copy_source)
        extra_args = call_args.extra_args
        for param, value in extra_args.items():
            if param in self.EXTRA_ARGS_TO_HEAD_ARGS_MAPPING:
                head_object_request[self.EXTRA_ARGS_TO_HEAD_ARGS_MAPPING[param]] = value
        response = call_args.source_client.head_object(**head_object_request)
        transfer_future.meta.provide_transfer_size(response['ContentLength'])
    if transfer_future.meta.size < config.multipart_threshold:
        self._submit_copy_request(client, config, osutil, request_executor, transfer_future)
    else:
        self._submit_multipart_request(client, config, osutil, request_executor, transfer_future)