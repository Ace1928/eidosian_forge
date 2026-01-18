from boto.compat import json
from boto.exception import JSONResponseError
from boto.connection import AWSAuthConnection
from boto.regioninfo import RegionInfo
from boto.cognito.sync import exceptions
def unsubscribe_from_dataset(self, identity_pool_id, identity_id, dataset_name, device_id):
    """
        Unsubscribe from receiving notifications when a dataset is
        modified by another device.

        :type identity_pool_id: string
        :param identity_pool_id: A name-spaced GUID (for example, us-
            east-1:23EC4050-6AEA-7089-A2DD-08002EXAMPLE) created by Amazon
            Cognito. The ID of the pool to which this identity belongs.

        :type identity_id: string
        :param identity_id: Unique ID for this identity.

        :type dataset_name: string
        :param dataset_name: The name of the dataset from which to unsubcribe.

        :type device_id: string
        :param device_id: The unique ID generated for this device by Cognito.

        """
    uri = '/identitypools/{0}/identities/{1}/datasets/{2}/subscriptions/{3}'.format(identity_pool_id, identity_id, dataset_name, device_id)
    return self.make_request('DELETE', uri, expected_status=200)