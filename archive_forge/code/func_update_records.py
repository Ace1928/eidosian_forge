from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cognito.sync import exceptions
def update_records(self, identity_pool_id, identity_id, dataset_name, sync_session_token, device_id=None, record_patches=None, client_context=None):
    """
        Posts updates to records and add and delete records for a
        dataset and user. The credentials used to make this API call
        need to have access to the identity data. With Amazon Cognito
        Sync, each identity has access only to its own data. You
        should use Amazon Cognito Identity service to retrieve the
        credentials necessary to make this API call.

        :type identity_pool_id: string
        :param identity_pool_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. GUID generation is unique within a region.

        :type identity_id: string
        :param identity_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. GUID generation is unique within a region.

        :type dataset_name: string
        :param dataset_name: A string of up to 128 characters. Allowed
            characters are a-z, A-Z, 0-9, '_' (underscore), '-' (dash), and '.'
            (dot).

        :type device_id: string
        :param device_id: The unique ID generated for this device by Cognito.

        :type record_patches: list
        :param record_patches: A list of patch operations.

        :type sync_session_token: string
        :param sync_session_token: The SyncSessionToken returned by a previous
            call to ListRecords for this dataset and identity.

        :type client_context: string
        :param client_context: Intended to supply a device ID that will
            populate the `lastModifiedBy` field referenced in other methods.
            The `ClientContext` field is not yet implemented.

        """
    uri = '/identitypools/{0}/identities/{1}/datasets/{2}'.format(identity_pool_id, identity_id, dataset_name)
    params = {'SyncSessionToken': sync_session_token}
    headers = {}
    query_params = {}
    if device_id is not None:
        params['DeviceId'] = device_id
    if record_patches is not None:
        params['RecordPatches'] = record_patches
    if client_context is not None:
        headers['x-amz-Client-Context'] = client_context
    if client_context is not None:
        headers['x-amz-Client-Context'] = client_context
    return self.make_request('POST', uri, expected_status=200, data=json.dumps(params), headers=headers, params=query_params)