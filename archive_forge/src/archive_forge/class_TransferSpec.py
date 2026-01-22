from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TransferSpec(_messages.Message):
    """Configuration for running a transfer.

  Fields:
    awsS3CompatibleDataSource: An AWS S3 compatible data source.
    awsS3DataSource: An AWS S3 data source.
    azureBlobStorageDataSource: An Azure Blob Storage data source.
    gcsDataSink: A Cloud Storage data sink.
    gcsDataSource: A Cloud Storage data source.
    gcsIntermediateDataLocation: For transfers between file systems, specifies
      a Cloud Storage bucket to be used as an intermediate location through
      which to transfer data. See [Transfer data between file
      systems](https://cloud.google.com/storage-transfer/docs/file-to-file)
      for more information.
    hdfsDataSource: An HDFS cluster data source.
    httpDataSource: An HTTP URL data source.
    objectConditions: Only objects that satisfy these object conditions are
      included in the set of data source and data sink objects. Object
      conditions based on objects' "last modification time" do not exclude
      objects in a data sink.
    posixDataSink: A POSIX Filesystem data sink.
    posixDataSource: A POSIX Filesystem data source.
    sinkAgentPoolName: Specifies the agent pool name associated with the posix
      data sink. When unspecified, the default name is used.
    sourceAgentPoolName: Specifies the agent pool name associated with the
      posix data source. When unspecified, the default name is used.
    transferManifest: A manifest file provides a list of objects to be
      transferred from the data source. This field points to the location of
      the manifest file. Otherwise, the entire source bucket is used.
      ObjectConditions still apply.
    transferOptions: If the option delete_objects_unique_in_sink is `true` and
      time-based object conditions such as 'last modification time' are
      specified, the request fails with an INVALID_ARGUMENT error.
  """
    awsS3CompatibleDataSource = _messages.MessageField('AwsS3CompatibleData', 1)
    awsS3DataSource = _messages.MessageField('AwsS3Data', 2)
    azureBlobStorageDataSource = _messages.MessageField('AzureBlobStorageData', 3)
    gcsDataSink = _messages.MessageField('GcsData', 4)
    gcsDataSource = _messages.MessageField('GcsData', 5)
    gcsIntermediateDataLocation = _messages.MessageField('GcsData', 6)
    hdfsDataSource = _messages.MessageField('HdfsData', 7)
    httpDataSource = _messages.MessageField('HttpData', 8)
    objectConditions = _messages.MessageField('ObjectConditions', 9)
    posixDataSink = _messages.MessageField('PosixFilesystem', 10)
    posixDataSource = _messages.MessageField('PosixFilesystem', 11)
    sinkAgentPoolName = _messages.StringField(12)
    sourceAgentPoolName = _messages.StringField(13)
    transferManifest = _messages.MessageField('TransferManifest', 14)
    transferOptions = _messages.MessageField('TransferOptions', 15)