from libcloud.utils.py3 import httplib, parse_qs, urlparse
from libcloud.common.base import BaseDriver, JsonResponse, ConnectionKey
from libcloud.common.types import LibcloudError, InvalidCredsError
class DigitalOcean_v2_BaseDriver(DigitalOceanBaseDriver):
    """
    DigitalOcean BaseDriver using v2 of the API.

    Supports `ex_per_page` ``int`` value keyword parameter to adjust per page
    requests against the API.
    """
    connectionCls = DigitalOcean_v2_Connection

    def __init__(self, key, secret=None, secure=True, host=None, port=None, api_version=None, region=None, ex_per_page=200, **kwargs):
        self.ex_per_page = ex_per_page
        super().__init__(key, **kwargs)

    def ex_account_info(self):
        return self.connection.request('/v2/account').object['account']

    def ex_list_events(self):
        return self._paginated_request('/v2/actions', 'actions')

    def ex_get_event(self, event_id):
        """
        Get an event object

        :param      event_id: Event id (required)
        :type       event_id: ``str``
        """
        params = {}
        return self.connection.request('/v2/actions/%s' % event_id, params=params).object['action']

    def _paginated_request(self, url, obj):
        """
        Perform multiple calls in order to have a full list of elements when
        the API responses are paginated.

        :param url: API endpoint
        :type url: ``str``

        :param obj: Result object key
        :type obj: ``str``

        :return: ``list`` of API response objects
        :rtype: ``list``
        """
        params = {}
        data = self.connection.request(url)
        try:
            query = urlparse.urlparse(data.object['links']['pages']['last'])
            pages = parse_qs(query[4])['page'][0]
            values = data.object[obj]
            for page in range(2, int(pages) + 1):
                params.update({'page': page})
                new_data = self.connection.request(url, params=params)
                more_values = new_data.object[obj]
                for value in more_values:
                    values.append(value)
            data = values
        except KeyError:
            data = data.object[obj]
        return data