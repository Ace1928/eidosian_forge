from botocore.exceptions import ClientError
from boto3.s3.transfer import create_transfer_manager
from boto3.s3.transfer import TransferConfig, S3Transfer
from boto3.s3.transfer import ProgressCallbackInvoker
from boto3 import utils
def upload_fileobj(self, Fileobj, Bucket, Key, ExtraArgs=None, Callback=None, Config=None):
    """Upload a file-like object to S3.

    The file-like object must be in binary mode.

    This is a managed transfer which will perform a multipart upload in
    multiple threads if necessary.

    Usage::

        import boto3
        s3 = boto3.client('s3')

        with open('filename', 'rb') as data:
            s3.upload_fileobj(data, 'mybucket', 'mykey')

    :type Fileobj: a file-like object
    :param Fileobj: A file-like object to upload. At a minimum, it must
        implement the `read` method, and must return bytes.

    :type Bucket: str
    :param Bucket: The name of the bucket to upload to.

    :type Key: str
    :param Key: The name of the key to upload to.

    :type ExtraArgs: dict
    :param ExtraArgs: Extra arguments that may be passed to the
        client operation.

    :type Callback: function
    :param Callback: A method which takes a number of bytes transferred to
        be periodically called during the upload.

    :type Config: boto3.s3.transfer.TransferConfig
    :param Config: The transfer configuration to be used when performing the
        upload.
    """
    if not hasattr(Fileobj, 'read'):
        raise ValueError('Fileobj must implement read')
    subscribers = None
    if Callback is not None:
        subscribers = [ProgressCallbackInvoker(Callback)]
    config = Config
    if config is None:
        config = TransferConfig()
    with create_transfer_manager(self, config) as manager:
        future = manager.upload(fileobj=Fileobj, bucket=Bucket, key=Key, extra_args=ExtraArgs, subscribers=subscribers)
        return future.result()