import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def upload_archive(self, vault_name, archive, linear_hash, tree_hash, description=None):
    """
        This operation adds an archive to a vault. This is a
        synchronous operation, and for a successful upload, your data
        is durably persisted. Amazon Glacier returns the archive ID in
        the `x-amz-archive-id` header of the response.

        You must use the archive ID to access your data in Amazon
        Glacier. After you upload an archive, you should save the
        archive ID returned so that you can retrieve or delete the
        archive later. Besides saving the archive ID, you can also
        index it and give it a friendly name to allow for better
        searching. You can also use the optional archive description
        field to specify how the archive is referred to in an external
        index of archives, such as you might create in Amazon
        DynamoDB. You can also get the vault inventory to obtain a
        list of archive IDs in a vault. For more information, see
        InitiateJob.

        You must provide a SHA256 tree hash of the data you are
        uploading. For information about computing a SHA256 tree hash,
        see `Computing Checksums`_.

        You can optionally specify an archive description of up to
        1,024 printable ASCII characters. You can get the archive
        description when you either retrieve the archive or get the
        vault inventory. For more information, see InitiateJob. Amazon
        Glacier does not interpret the description in any way. An
        archive description does not need to be unique. You cannot use
        the description to retrieve or sort the archive list.

        Archives are immutable. After you upload an archive, you
        cannot edit the archive or its description.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and underlying REST API, go to
        `Uploading an Archive in Amazon Glacier`_ and `Upload
        Archive`_ in the Amazon Glacier Developer Guide .

        :type vault_name: str
        :param vault_name: The name of the vault

        :type archive: bytes
        :param archive: The data to upload.

        :type linear_hash: str
        :param linear_hash: The SHA256 checksum (a linear hash) of the
            payload.

        :type tree_hash: str
        :param tree_hash: The user-computed SHA256 tree hash of the
            payload.  For more information on computing the
            tree hash, see http://goo.gl/u7chF.

        :type description: str
        :param description: The optional description of the archive you
            are uploading.
        """
    response_headers = [('x-amz-archive-id', u'ArchiveId'), ('Location', u'Location'), ('x-amz-sha256-tree-hash', u'TreeHash')]
    uri = 'vaults/%s/archives' % vault_name
    try:
        content_length = str(len(archive))
    except (TypeError, AttributeError):
        content_length = str(os.fstat(archive.fileno()).st_size)
    headers = {'x-amz-content-sha256': linear_hash, 'x-amz-sha256-tree-hash': tree_hash, 'Content-Length': content_length}
    if description:
        headers['x-amz-archive-description'] = description
    if self._is_file_like(archive):
        sender = ResettingFileSender(archive)
    else:
        sender = None
    return self.make_request('POST', uri, headers=headers, sender=sender, data=archive, ok_responses=(201,), response_headers=response_headers)