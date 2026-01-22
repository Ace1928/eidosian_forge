import re
from . import urlutils
from .hooks import Hooks
class LocationHooks(Hooks):
    """Dictionary mapping hook name to a list of callables for location hooks.
    """

    def __init__(self):
        Hooks.__init__(self, 'breezy.location', 'hooks')
        self.add_hook('rewrite_url', 'Possibly rewrite a URL. Called with a URL to rewrite and the purpose of the URL.', (3, 0))
        self.add_hook('rewrite_location', 'Possibly rewrite a location. Called with a location string to rewrite and the purpose of the URL.', (3, 2))