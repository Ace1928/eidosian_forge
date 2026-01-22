import abc
import copy
from http import client as http_client
from urllib import parse as urlparse
from oslo_utils import strutils
from ironicclient.common.apiclient import exceptions
from ironicclient.common.i18n import _
class CrudManager(BaseManager):
    """Base manager class for manipulating entities.

    Children of this class are expected to define a `collection_key` and `key`.

    - `collection_key`: Usually a plural noun by convention (e.g. `entities`);
      used to refer collections in both URL's (e.g.  `/v3/entities`) and JSON
      objects containing a list of member resources (e.g. `{'entities': [{},
      {}, {}]}`).
    - `key`: Usually a singular noun by convention (e.g. `entity`); used to
      refer to an individual member of the collection.

    """
    collection_key = None
    key = None

    def build_url(self, base_url=None, **kwargs):
        """Builds a resource URL for the given kwargs.

        Given an example collection where `collection_key = 'entities'` and
        `key = 'entity'`, the following URL's could be generated.

        By default, the URL will represent a collection of entities, e.g.::

            /entities

        If kwargs contains an `entity_id`, then the URL will represent a
        specific member, e.g.::

            /entities/{entity_id}

        :param base_url: if provided, the generated URL will be appended to it
        """
        url = base_url if base_url is not None else ''
        url += '/%s' % self.collection_key
        entity_id = kwargs.get('%s_id' % self.key)
        if entity_id is not None:
            url += '/%s' % entity_id
        return url

    def _filter_kwargs(self, kwargs):
        """Drop null values and handle ids."""
        for key, ref in kwargs.copy().items():
            if ref is None:
                kwargs.pop(key)
            elif isinstance(ref, Resource):
                kwargs.pop(key)
                kwargs['%s_id' % key] = getid(ref)
        return kwargs

    def create(self, **kwargs):
        kwargs = self._filter_kwargs(kwargs)
        return self._post(self.build_url(**kwargs), {self.key: kwargs}, self.key)

    def get(self, **kwargs):
        kwargs = self._filter_kwargs(kwargs)
        return self._get(self.build_url(**kwargs), self.key)

    def head(self, **kwargs):
        kwargs = self._filter_kwargs(kwargs)
        return self._head(self.build_url(**kwargs))

    def list(self, base_url=None, **kwargs):
        """List the collection.

        :param base_url: if provided, the generated URL will be appended to it
        """
        kwargs = self._filter_kwargs(kwargs)
        return self._list('%(base_url)s%(query)s' % {'base_url': self.build_url(base_url=base_url, **kwargs), 'query': '?%s' % urlparse.urlencode(kwargs) if kwargs else ''}, self.collection_key)

    def put(self, base_url=None, **kwargs):
        """Update an element.

        :param base_url: if provided, the generated URL will be appended to it
        """
        kwargs = self._filter_kwargs(kwargs)
        return self._put(self.build_url(base_url=base_url, **kwargs))

    def update(self, **kwargs):
        kwargs = self._filter_kwargs(kwargs)
        params = kwargs.copy()
        params.pop('%s_id' % self.key)
        return self._patch(self.build_url(**kwargs), {self.key: params}, self.key)

    def delete(self, **kwargs):
        kwargs = self._filter_kwargs(kwargs)
        return self._delete(self.build_url(**kwargs))

    def find(self, base_url=None, **kwargs):
        """Find a single item with attributes matching ``**kwargs``.

        :param base_url: if provided, the generated URL will be appended to it
        """
        kwargs = self._filter_kwargs(kwargs)
        rl = self._list('%(base_url)s%(query)s' % {'base_url': self.build_url(base_url=base_url, **kwargs), 'query': '?%s' % urlparse.urlencode(kwargs) if kwargs else ''}, self.collection_key)
        num = len(rl)
        if num == 0:
            msg = _('No %(name)s matching %(args)s.') % {'name': self.resource_class.__name__, 'args': kwargs}
            raise exceptions.NotFound(msg)
        elif num > 1:
            raise exceptions.NoUniqueMatch
        else:
            return rl[0]