import boto
from boto.compat import json
from boto.connection import AWSQueryConnection
from boto.regioninfo import RegionInfo
from boto.exception import JSONResponseError
from boto.cognito.identity import exceptions
def unlink_identity(self, identity_id, logins, logins_to_remove):
    """
        Unlinks a federated identity from an existing account.
        Unlinked logins will be considered new identities next time
        they are seen. Removing the last linked login will make this
        identity inaccessible.

        :type identity_id: string
        :param identity_id: A unique identifier in the format REGION:GUID.

        :type logins: map
        :param logins: A set of optional name-value pairs that map provider
            names to provider tokens.

        :type logins_to_remove: list
        :param logins_to_remove: Provider names to unlink from this identity.

        """
    params = {'IdentityId': identity_id, 'Logins': logins, 'LoginsToRemove': logins_to_remove}
    return self.make_request(action='UnlinkIdentity', body=json.dumps(params))