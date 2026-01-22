import copy
import logging
from s3transfer.utils import get_callbacks
class CompleteMultipartUploadTask(Task):
    """Task to complete a multipart upload"""

    def _main(self, client, bucket, key, upload_id, parts, extra_args):
        """
        :param client: The client to use when calling CompleteMultipartUpload
        :param bucket: The name of the bucket to upload to
        :param key: The name of the key to upload to
        :param upload_id: The id of the upload
        :param parts: A list of parts to use to complete the multipart upload::

            [{'Etag': etag_value, 'PartNumber': part_number}, ...]

            Each element in the list consists of a return value from
            ``UploadPartTask.main()``.
        :param extra_args:  A dictionary of any extra arguments that may be
            used in completing the multipart transfer.
        """
        client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id, MultipartUpload={'Parts': parts}, **extra_args)