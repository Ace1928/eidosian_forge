import os
import boto.glacier
from boto.compat import json
from boto.connection import AWSAuthConnection
from boto.glacier.exceptions import UnexpectedHTTPResponseError
from boto.glacier.response import GlacierResponse
from boto.glacier.utils import ResettingFileSender
def abort_multipart_upload(self, vault_name, upload_id):
    """
        This operation aborts a multipart upload identified by the
        upload ID.

        After the Abort Multipart Upload request succeeds, you cannot
        upload any more parts to the multipart upload or complete the
        multipart upload. Aborting a completed upload fails. However,
        aborting an already-aborted upload will succeed, for a short
        time. For more information about uploading a part and
        completing a multipart upload, see UploadMultipartPart and
        CompleteMultipartUpload.

        This operation is idempotent.

        An AWS account has full permission to perform all operations
        (actions). However, AWS Identity and Access Management (IAM)
        users don't have any permissions by default. You must grant
        them explicit permission to perform specific actions. For more
        information, see `Access Control Using AWS Identity and Access
        Management (IAM)`_.

        For conceptual information and underlying REST API, go to
        `Working with Archives in Amazon Glacier`_ and `Abort
        Multipart Upload`_ in the Amazon Glacier Developer Guide .

        :type vault_name: string
        :param vault_name: The name of the vault.

        :type upload_id: string
        :param upload_id: The upload ID of the multipart upload to delete.
        """
    uri = 'vaults/%s/multipart-uploads/%s' % (vault_name, upload_id)
    return self.make_request('DELETE', uri, ok_responses=(204,))