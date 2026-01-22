from boto.compat import unquote_str
class MultiPartUploadListResultSet(object):
    """
    A resultset for listing multipart uploads within a bucket.
    Uses the multipart_upload_lister generator function and
    implements the iterator interface.  This
    transparently handles the results paging from S3 so even if you have
    many thousands of uploads within the bucket you can iterate over all
    keys in a reasonably efficient manner.
    """

    def __init__(self, bucket=None, key_marker='', upload_id_marker='', headers=None, encoding_type=None):
        self.bucket = bucket
        self.key_marker = key_marker
        self.upload_id_marker = upload_id_marker
        self.headers = headers
        self.encoding_type = encoding_type

    def __iter__(self):
        return multipart_upload_lister(self.bucket, key_marker=self.key_marker, upload_id_marker=self.upload_id_marker, headers=self.headers, encoding_type=self.encoding_type)