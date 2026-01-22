from lazr.restfulclient.resource import (
class CookbookSet(CollectionWithKeyBasedLookup):
    """A custom subclass capable of cookbook lookup by cookbook name."""

    def _get_url_from_id(self, id):
        """Transform a cookbook name into the URL to a cookbook resource."""
        return str(self._root._root_uri.ensureSlash()) + 'cookbooks/' + quote(str(id))
    collection_of = 'cookbook'