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
class UploadSeekableInputManager(UploadFilenameInputManager):
    """Upload utility for an open file object"""

    @classmethod
    def is_compatible(cls, upload_source):
        return readable(upload_source) and seekable(upload_source)

    def stores_body_in_memory(self, operation_name):
        if operation_name == 'put_object':
            return False
        else:
            return True

    def provide_transfer_size(self, transfer_future):
        fileobj = transfer_future.meta.call_args.fileobj
        start_position = fileobj.tell()
        fileobj.seek(0, 2)
        end_position = fileobj.tell()
        fileobj.seek(start_position)
        transfer_future.meta.provide_transfer_size(end_position - start_position)

    def _get_upload_part_fileobj_with_full_size(self, fileobj, **kwargs):
        data = fileobj.read(kwargs['part_size'])
        return (six.BytesIO(data), len(data))

    def _get_put_object_fileobj_with_full_size(self, transfer_future):
        fileobj = transfer_future.meta.call_args.fileobj
        size = fileobj.tell() + transfer_future.meta.size
        return (fileobj, size)