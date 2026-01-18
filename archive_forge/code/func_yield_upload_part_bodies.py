import math
from botocore.compat import six
from s3transfer.compat import seekable, readable
from s3transfer.futures import IN_MEMORY_UPLOAD_TAG
from s3transfer.tasks import Task
from s3transfer.tasks import SubmissionTask
from s3transfer.tasks import CreateMultipartUploadTask
from s3transfer.tasks import CompleteMultipartUploadTask
from s3transfer.utils import get_callbacks
from s3transfer.utils import get_filtered_dict
from s3transfer.utils import DeferredOpenFile, ChunksizeAdjuster
def yield_upload_part_bodies(self, transfer_future, chunksize):
    file_object = transfer_future.meta.call_args.fileobj
    part_number = 0
    while True:
        callbacks = self._get_progress_callbacks(transfer_future)
        close_callbacks = self._get_close_callbacks(callbacks)
        part_number += 1
        part_content = self._read(file_object, chunksize)
        if not part_content:
            break
        part_object = self._wrap_data(part_content, callbacks, close_callbacks)
        part_content = None
        yield (part_number, part_object)