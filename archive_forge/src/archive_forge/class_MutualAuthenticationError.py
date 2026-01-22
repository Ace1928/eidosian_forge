from requests.exceptions import RequestException
class MutualAuthenticationError(RequestException):
    """Mutual Authentication Error"""