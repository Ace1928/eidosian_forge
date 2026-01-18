import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions

        Updates a user pool.

        :type identity_pool_id: string
        :param identity_pool_id: An identity pool ID in the format REGION:GUID.

        :type identity_pool_name: string
        :param identity_pool_name: A string that you provide.

        :type allow_unauthenticated_identities: boolean
        :param allow_unauthenticated_identities: TRUE if the identity pool
            supports unauthenticated logins.

        :type supported_login_providers: map
        :param supported_login_providers: Optional key:value pairs mapping
            provider names to provider app IDs.

        :type developer_provider_name: string
        :param developer_provider_name: The "domain" by which Cognito will
            refer to your users.

        :type open_id_connect_provider_ar_ns: list
        :param open_id_connect_provider_ar_ns:

        