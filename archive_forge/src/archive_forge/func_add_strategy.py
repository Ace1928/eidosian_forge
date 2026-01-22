from requests.auth import AuthBase, HTTPBasicAuth
from requests.compat import urlparse, urlunparse
def add_strategy(self, domain, strategy):
    """Add a new domain and authentication strategy.

        :param str domain: The domain you wish to match against. For example:
            ``'https://api.github.com'``
        :param str strategy: The authentication strategy you wish to use for
            that domain. For example: ``('username', 'password')`` or
            ``requests.HTTPDigestAuth('username', 'password')``

        .. code-block:: python

            a = AuthHandler({})
            a.add_strategy('https://api.github.com', ('username', 'password'))

        """
    if isinstance(strategy, tuple):
        strategy = HTTPBasicAuth(*strategy)
    key = self._key_from_url(domain)
    self.strategies[key] = strategy