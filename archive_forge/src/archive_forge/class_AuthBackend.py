import abc
class AuthBackend(object, metaclass=abc.ABCMeta):

    def __init__(self, conf):
        self.conf = conf

    @abc.abstractmethod
    def authenticate(self, api_version, request):
        """Authenticates the user in the selected backend.

        Auth backends will have to manipulate the
        request and prepare it to send the auth information
        back to Zaqar's instance.

        :params api_version: Zaqar's API version.
        :params request: Request Spec instance
            that can be manipulated by the backend
            if the authentication succeeds.

        :returns: The modified request spec.
        """