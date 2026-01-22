from __future__ import (absolute_import, division, print_function)
from ansible.errors import AnsibleError
from ansible.plugins.lookup import LookupBase
from ansible.module_utils.urls import open_url, ConnectionError, SSLValidationError
from ansible.module_utils.six.moves.urllib.error import HTTPError, URLError
from ansible.module_utils.six.moves.urllib.parse import urlencode
from ansible.module_utils import six
from ansible.utils.display import Display
from traceback import format_exception
import json
import sys
class ManifoldApiClient(object):
    base_url = 'https://api.{api}.manifold.co/v1/{endpoint}'
    http_agent = 'python-manifold-ansible-1.0.0'

    def __init__(self, token):
        self._token = token

    def request(self, api, endpoint, *args, **kwargs):
        """
        Send a request to API backend and pre-process a response.
        :param api: API to send a request to
        :type api: str
        :param endpoint: API endpoint to fetch data from
        :type endpoint: str
        :param args: other args for open_url
        :param kwargs: other kwargs for open_url
        :return: server response. JSON response is automatically deserialized.
        :rtype: dict | list | str
        """
        default_headers = {'Authorization': 'Bearer {0}'.format(self._token), 'Accept': '*/*'}
        url = self.base_url.format(api=api, endpoint=endpoint)
        headers = default_headers
        arg_headers = kwargs.pop('headers', None)
        if arg_headers:
            headers.update(arg_headers)
        try:
            display.vvvv('manifold lookup connecting to {0}'.format(url))
            response = open_url(url, *args, headers=headers, http_agent=self.http_agent, **kwargs)
            data = response.read()
            if response.headers.get('content-type') == 'application/json':
                data = json.loads(data)
            return data
        except ValueError:
            raise ApiError("JSON response can't be parsed while requesting {url}:\n{json}".format(json=data, url=url))
        except HTTPError as e:
            raise ApiError('Server returned: {err} while requesting {url}:\n{response}'.format(err=str(e), url=url, response=e.read()))
        except URLError as e:
            raise ApiError('Failed lookup url for {url} : {err}'.format(url=url, err=str(e)))
        except SSLValidationError as e:
            raise ApiError("Error validating the server's certificate for {url}: {err}".format(url=url, err=str(e)))
        except ConnectionError as e:
            raise ApiError('Error connecting to {url}: {err}'.format(url=url, err=str(e)))

    def get_resources(self, team_id=None, project_id=None, label=None):
        """
        Get resources list
        :param team_id: ID of the Team to filter resources by
        :type team_id: str
        :param project_id: ID of the project to filter resources by
        :type project_id: str
        :param label: filter resources by a label, returns a list with one or zero elements
        :type label: str
        :return: list of resources
        :rtype: list
        """
        api = 'marketplace'
        endpoint = 'resources'
        query_params = {}
        if team_id:
            query_params['team_id'] = team_id
        if project_id:
            query_params['project_id'] = project_id
        if label:
            query_params['label'] = label
        if query_params:
            endpoint += '?' + urlencode(query_params)
        return self.request(api, endpoint)

    def get_teams(self, label=None):
        """
        Get teams list
        :param label: filter teams by a label, returns a list with one or zero elements
        :type label: str
        :return: list of teams
        :rtype: list
        """
        api = 'identity'
        endpoint = 'teams'
        data = self.request(api, endpoint)
        if label:
            data = list(filter(lambda x: x['body']['label'] == label, data))
        return data

    def get_projects(self, label=None):
        """
        Get projects list
        :param label: filter projects by a label, returns a list with one or zero elements
        :type label: str
        :return: list of projects
        :rtype: list
        """
        api = 'marketplace'
        endpoint = 'projects'
        query_params = {}
        if label:
            query_params['label'] = label
        if query_params:
            endpoint += '?' + urlencode(query_params)
        return self.request(api, endpoint)

    def get_credentials(self, resource_id):
        """
        Get resource credentials
        :param resource_id: ID of the resource to filter credentials by
        :type resource_id: str
        :return:
        """
        api = 'marketplace'
        endpoint = 'credentials?' + urlencode({'resource_id': resource_id})
        return self.request(api, endpoint)