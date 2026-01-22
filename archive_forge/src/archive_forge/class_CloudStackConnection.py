import copy
import hmac
import base64
import hashlib
from libcloud.utils.py3 import b, httplib, urlquote, urlencode
from libcloud.common.base import JsonResponse, PollingConnection, ConnectionUserAndKey
from libcloud.common.types import ProviderError, MalformedResponseError
from libcloud.compute.types import InvalidCredsError
class CloudStackConnection(ConnectionUserAndKey, PollingConnection):
    responseCls = CloudStackResponse
    poll_interval = 1
    request_method = '_sync_request'
    timeout = 600
    ASYNC_PENDING = 0
    ASYNC_SUCCESS = 1
    ASYNC_FAILURE = 2

    def encode_data(self, data):
        """
        Must of the data is sent as part of query params (eeww),
        but in newer versions, userdata argument can be sent as a
        urlencoded data in the request body.
        """
        if data:
            data = urlencode(data)
        return data

    def _make_signature(self, params):
        signature = [(k.lower(), v) for k, v in list(params.items())]
        signature.sort(key=lambda x: x[0])
        pairs = []
        for pair in signature:
            key = urlquote(str(pair[0]), safe='[]')
            value = urlquote(str(pair[1]), safe='[]*')
            item = '{}={}'.format(key, value)
            pairs.append(item)
        signature = '&'.join(pairs)
        signature = signature.lower().replace('+', '%20')
        signature = hmac.new(b(self.key), msg=b(signature), digestmod=hashlib.sha1)
        return base64.b64encode(b(signature.digest()))

    def add_default_params(self, params):
        params['apiKey'] = self.user_id
        params['response'] = 'json'
        return params

    def pre_connect_hook(self, params, headers):
        params['signature'] = self._make_signature(params)
        return (params, headers)

    def _async_request(self, command, action=None, params=None, data=None, headers=None, method='GET', context=None):
        if params:
            context = copy.deepcopy(params)
        else:
            context = {}
        context['command'] = command
        result = super().async_request(action=action, params=params, data=data, headers=headers, method=method, context=context)
        return result['jobresult']

    def get_request_kwargs(self, action, params=None, data='', headers=None, method='GET', context=None):
        command = context['command']
        request_kwargs = {'command': command, 'action': action, 'params': params, 'data': data, 'headers': headers, 'method': method}
        return request_kwargs

    def get_poll_request_kwargs(self, response, context, request_kwargs):
        job_id = response['jobid']
        params = {'jobid': job_id}
        kwargs = {'command': 'queryAsyncJobResult', 'params': params}
        return kwargs

    def has_completed(self, response):
        status = response.get('jobstatus', self.ASYNC_PENDING)
        if status == self.ASYNC_FAILURE:
            msg = response.get('jobresult', {}).get('errortext', status)
            raise Exception(msg)
        return status == self.ASYNC_SUCCESS

    def _sync_request(self, command, action=None, params=None, data=None, headers=None, method='GET'):
        """
        This method handles synchronous calls which are generally fast
        information retrieval requests and thus return 'quickly'.
        """
        if params:
            params = copy.deepcopy(params)
        else:
            params = {}
        params['command'] = command
        result = self.request(action=self.driver.path, params=params, data=data, headers=headers, method=method)
        command = command.lower()
        if command == 'revokesecuritygroupingress' and 'revokesecuritygroupingressresponse' not in result.object:
            command = command
        elif command == 'restorevirtualmachine' and 'restorevmresponse' in result.object:
            command = 'restorevmresponse'
        else:
            command = command + 'response'
        if command not in result.object:
            raise MalformedResponseError('Unknown response format {}'.format(command), body=result.body, driver=self.driver)
        result = result.object[command]
        return result