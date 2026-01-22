from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import local_file_uploader
class PySparkBatchFactory(object):
    """Factory class for PySparkBatch message."""

    def __init__(self, dataproc):
        """Factory class for SparkBatch message.

    Args:
      dataproc: A Dataproc instance.
    """
        self.dataproc = dataproc

    def UploadLocalFilesAndGetMessage(self, args):
        """upload user local files and creates a PySparkBatch message.

    Upload user local files and point URIs to the local files to the uploaded
    URIs.
    Creates a PySparkBatch message from parsed arguments.

    Args:
      args: Parsed arguments.

    Returns:
      PySparkBatch: A PySparkBatch message.

    Raises:
      AttributeError: Bucket is required to upload local files, but not
      specified.
    """
        kwargs = {}
        if args.args:
            kwargs['args'] = args.args
        dependencies = {}
        dependencies['mainPythonFileUri'] = [args.MAIN_PYTHON_FILE]
        if args.py_files:
            dependencies['pythonFileUris'] = args.py_files
        if args.jars:
            dependencies['jarFileUris'] = args.jars
        if args.files:
            dependencies['fileUris'] = args.files
        if args.archives:
            dependencies['archiveUris'] = args.archives
        if local_file_uploader.HasLocalFiles(dependencies):
            if not args.deps_bucket:
                raise AttributeError('--deps-bucket was not specified.')
            dependencies = local_file_uploader.Upload(args.deps_bucket, dependencies)
        dependencies['mainPythonFileUri'] = dependencies['mainPythonFileUri'][0]
        kwargs.update(dependencies)
        return self.dataproc.messages.PySparkBatch(**kwargs)