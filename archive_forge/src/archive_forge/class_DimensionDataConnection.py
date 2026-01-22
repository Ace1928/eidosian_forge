from time import sleep
from base64 import b64encode
from libcloud.utils.py3 import b, httplib, basestring
from libcloud.utils.xml import findtext
from libcloud.common.base import RawResponse, XmlResponse, ConnectionUserAndKey
from libcloud.compute.base import Node
from libcloud.compute.types import LibcloudError, InvalidCredsError
class DimensionDataConnection(ConnectionUserAndKey):
    """
    Connection class for the DimensionData driver
    """
    api_path_version_1 = '/oec'
    api_path_version_2 = '/caas'
    api_version_1 = 0.9
    oldest_api_version = '2.2'
    latest_api_version = '2.4'
    active_api_version = '2.4'
    _orgId = None
    responseCls = DimensionDataResponse
    rawResponseCls = DimensionDataRawResponse
    allow_insecure = False

    def __init__(self, user_id, key, secure=True, host=None, port=None, url=None, timeout=None, proxy_url=None, api_version=None, **conn_kwargs):
        super().__init__(user_id=user_id, key=key, secure=secure, host=host, port=port, url=url, timeout=timeout, proxy_url=proxy_url)
        if conn_kwargs['region']:
            self.host = conn_kwargs['region']['host']
        if api_version:
            if LooseVersion(api_version) < LooseVersion(self.oldest_api_version):
                msg = 'API Version specified is too old. No longer supported. Please upgrade to the latest version {}'.format(self.active_api_version)
                raise DimensionDataAPIException(code=None, msg=msg, driver=self.driver)
            elif LooseVersion(api_version) > LooseVersion(self.latest_api_version):
                msg = 'Unsupported API Version. The version specified is not release yet. Please use the latest supported version {}'.format(self.active_api_version)
                raise DimensionDataAPIException(code=None, msg=msg, driver=self.driver)
            else:
                self.active_api_version = api_version

    def add_default_headers(self, headers):
        headers['Authorization'] = 'Basic %s' % b64encode(b('{}:{}'.format(self.user_id, self.key))).decode('utf-8')
        headers['Content-Type'] = 'application/xml'
        return headers

    def request_api_1(self, action, params=None, data='', headers=None, method='GET'):
        action = '{}/{}/{}'.format(self.api_path_version_1, self.api_version_1, action)
        return super().request(action=action, params=params, data=data, method=method, headers=headers)

    def request_api_2(self, path, action, params=None, data='', headers=None, method='GET'):
        action = '{}/{}/{}/{}'.format(self.api_path_version_2, self.active_api_version, path, action)
        return super().request(action=action, params=params, data=data, method=method, headers=headers)

    def raw_request_with_orgId_api_1(self, action, params=None, data='', headers=None, method='GET'):
        action = '{}/{}'.format(self.get_resource_path_api_1(), action)
        return super().request(action=action, params=params, data=data, method=method, headers=headers, raw=True)

    def request_with_orgId_api_1(self, action, params=None, data='', headers=None, method='GET'):
        action = '{}/{}'.format(self.get_resource_path_api_1(), action)
        return super().request(action=action, params=params, data=data, method=method, headers=headers)

    def request_with_orgId_api_2(self, action, params=None, data='', headers=None, method='GET'):
        action = '{}/{}'.format(self.get_resource_path_api_2(), action)
        return super().request(action=action, params=params, data=data, method=method, headers=headers)

    def paginated_request_with_orgId_api_2(self, action, params=None, data='', headers=None, method='GET', page_size=250):
        """
        A paginated request to the MCP2.0 API
        This essentially calls out to request_with_orgId_api_2 for each page
        and yields the response to make a generator
        This generator can be looped through to grab all the pages.

        :param action: The resource to access (i.e. 'network/vlan')
        :type  action: ``str``

        :param params: Parameters to give to the action
        :type  params: ``dict`` or ``None``

        :param data: The data payload to be added to the request
        :type  data: ``str``

        :param headers: Additional header to be added to the request
        :type  headers: ``str`` or ``dict`` or ``None``

        :param method: HTTP Method for the request (i.e. 'GET', 'POST')
        :type  method: ``str``

        :param page_size: The size of each page to be returned
                          Note: Max page size in MCP2.0 is currently 250
        :type  page_size: ``int``
        """
        if params is None:
            params = {}
        params['pageSize'] = page_size
        resp = self.request_with_orgId_api_2(action, params, data, headers, method).object
        yield resp
        if len(resp) <= 0:
            return
        pcount = resp.get('pageCount')
        psize = resp.get('pageSize')
        pnumber = resp.get('pageNumber')
        while int(pcount) >= int(psize):
            params['pageNumber'] = int(pnumber) + 1
            resp = self.request_with_orgId_api_2(action, params, data, headers, method).object
            pcount = resp.get('pageCount')
            psize = resp.get('pageSize')
            pnumber = resp.get('pageNumber')
            yield resp

    def get_resource_path_api_1(self):
        """
        This method returns a resource path which is necessary for referencing
        resources that require a full path instead of just an ID, such as
        networks, and customer snapshots.
        """
        return '{}/{}/{}'.format(self.api_path_version_1, self.api_version_1, self._get_orgId())

    def get_resource_path_api_2(self):
        """
        This method returns a resource path which is necessary for referencing
        resources that require a full path instead of just an ID, such as
        networks, and customer snapshots.
        """
        return '{}/{}/{}'.format(self.api_path_version_2, self.active_api_version, self._get_orgId())

    def wait_for_state(self, state, func, poll_interval=2, timeout=60, *args, **kwargs):
        """
        Wait for the function which returns a instance with field status/state
        to match.

        Keep polling func until one of the desired states is matched

        :param state: Either the desired state (`str`) or a `list` of states
        :type  state: ``str`` or ``list``

        :param  func: The function to call, e.g. ex_get_vlan. Note: This
                      function needs to return an object which has ``status``
                      attribute.
        :type   func: ``function``

        :param  poll_interval: The number of seconds to wait between checks
        :type   poll_interval: `int`

        :param  timeout: The total number of seconds to wait to reach a state
        :type   timeout: `int`

        :param  args: The arguments for func
        :type   args: Positional arguments

        :param  kwargs: The arguments for func
        :type   kwargs: Keyword arguments

        :return: Result from the calling function.
        """
        cnt = 0
        result = None
        object_state = None
        while cnt < timeout / poll_interval:
            result = func(*args, **kwargs)
            if isinstance(result, Node):
                object_state = result.state
            else:
                object_state = result.status
            if object_state is state or str(object_state) in state:
                return result
            sleep(poll_interval)
            cnt += 1
        msg = 'Status check for object %s timed out' % result
        raise DimensionDataAPIException(code=object_state, msg=msg, driver=self.driver)

    def _get_orgId(self):
        """
        Send the /myaccount API request to DimensionData cloud and parse the
        'orgId' from the XML response object. We need the orgId to use most
        of the other API functions
        """
        if self._orgId is None:
            body = self.request_api_1('myaccount').object
            self._orgId = findtext(body, 'orgId', DIRECTORY_NS)
        return self._orgId

    def get_account_details(self):
        """
        Get the details of this account

        :rtype: :class:`DimensionDataAccountDetails`
        """
        body = self.request_api_1('myaccount').object
        return DimensionDataAccountDetails(user_name=findtext(body, 'userName', DIRECTORY_NS), full_name=findtext(body, 'fullName', DIRECTORY_NS), first_name=findtext(body, 'firstName', DIRECTORY_NS), last_name=findtext(body, 'lastName', DIRECTORY_NS), email=findtext(body, 'emailAddress', DIRECTORY_NS))