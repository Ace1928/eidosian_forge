from requests.auth import AuthBase, HTTPBasicAuth
from requests.compat import urlparse, urlunparse
def _make_uniform(self):
    existing_strategies = list(self.strategies.items())
    self.strategies = {}
    for k, v in existing_strategies:
        self.add_strategy(k, v)