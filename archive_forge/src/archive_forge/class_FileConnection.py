from boto.file.bucket import Bucket
class FileConnection(object):

    def __init__(self, file_storage_uri):
        self.file_storage_uri = file_storage_uri

    def get_bucket(self, bucket_name, validate=True, headers=None):
        return Bucket(bucket_name, self.file_storage_uri.object_name)