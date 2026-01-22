from zaqarclient.auth import base
class SignedURLAuth(base.AuthBackend):
    """Authenticate using signature.

    The returned client will only work on one dedicated queue which has been
    signed.

    :params conf: A dictionary with the signed URL data:
            - expires
            - methods
            - paths
            - signature
            - os_project_id
    :type conf: `dict`
    """

    def authenticate(self, api_version, request):
        """Set the necessary headers on the request."""
        request.headers['URL-Expires'] = self.conf['expires']
        request.headers['URL-Methods'] = ','.join(self.conf['methods'])
        request.headers['URL-Paths'] = ','.join(self.conf['paths'])
        request.headers['URL-Signature'] = self.conf['signature']
        return request