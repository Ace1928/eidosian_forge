from zunclient.common import base
from zunclient.common import utils
class HostManager(base.Manager):
    resource_class = Host

    @staticmethod
    def _path(id=None):
        if id:
            return '/v1/hosts/%s' % id
        else:
            return '/v1/hosts/'

    def list(self, marker=None, limit=None, sort_key=None, sort_dir=None):
        """Retrieve a list of hosts.

        :param marker: Optional, the UUID of an host, eg the last
                       host from a previous result set. Return
                       the next result set.
        :param limit: The maximum number of results to return per
                      request, if:

            1) limit > 0, the maximum number of hosts to return.
            2) limit param is NOT specified (None), the number of items
               returned respect the maximum imposed by the Zun api

        :param sort_key: Optional, field used for sorting.

        :param sort_dir: Optional, direction of sorting, either 'asc' (the
                         default) or 'desc'.

        :returns: A list of hosts.

        """
        if limit is not None:
            limit = int(limit)
        filters = utils.common_filters(marker, limit, sort_key, sort_dir)
        path = ''
        if filters:
            path += '?' + '&'.join(filters)
        if limit is None:
            return self._list(self._path(path), 'hosts')
        else:
            return self._list_pagination(self._path(path), 'hosts', limit=limit)

    def get(self, id):
        try:
            return self._list(self._path(id))[0]
        except IndexError:
            return None