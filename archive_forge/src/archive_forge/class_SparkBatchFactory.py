from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.dataproc import flags
from googlecloudsdk.command_lib.dataproc import local_file_uploader
class SparkBatchFactory(object):
    """Factory class for SparkBatch message."""

    def __init__(self, dataproc):
        """Factory class for SparkBatch message.

    Args:
      dataproc: A Dataproc instance.
    """
        self.dataproc = dataproc

    def UploadLocalFilesAndGetMessage(self, args):
        """Uploads local files and creates a SparkBatch message.

    Uploads user local files and change the URIs to local files to point to
    uploaded URIs.
    Creates a SparkBatch message from parsed arguments.

    Args:
      args: Parsed arguments.

    Returns:
      SparkBatch: A SparkBatch message.

    Raises:
      AttributeError: Main class and jar are missing, or both were provided.
      Bucket is required to upload local files, but not specified.
    """
        kwargs = {}
        if args.args:
            kwargs['args'] = args.args
        if not args.main_class and (not args.main_jar):
            raise AttributeError('Missing JVM main.')
        if args.main_class and args.main_jar:
            raise AttributeError("Can't provide both main class and jar.")
        dependencies = {}
        if args.main_class:
            kwargs['mainClass'] = args.main_class
        else:
            dependencies['mainJarFileUri'] = [args.main_jar]
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
        if 'mainJarFileUri' in dependencies:
            dependencies['mainJarFileUri'] = dependencies['mainJarFileUri'][0]
        kwargs.update(dependencies)
        return self.dataproc.messages.SparkBatch(**kwargs)