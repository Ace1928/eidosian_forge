from libcloud.utils.py3 import b, httplib, base64_encode_string
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.compute.types import InvalidCredsError
class BrightboxConnection(ConnectionUserAndKey):
    """
    Connection class for the Brightbox driver
    """
    host = 'api.gb1.brightbox.com'
    responseCls = BrightboxResponse

    def _fetch_oauth_token(self):
        body = json.dumps({'client_id': self.user_id, 'grant_type': 'none'})
        authorization = 'Basic ' + str(base64_encode_string(b('{}:{}'.format(self.user_id, self.key)))).rstrip()
        self.connect()
        headers = {'Host': self.host, 'User-Agent': self._user_agent(), 'Authorization': authorization, 'Content-Type': 'application/json', 'Content-Length': str(len(body))}
        response = self.connection.request(method='POST', url='/token', body=body, headers=headers)
        if response.status == httplib.OK:
            return json.loads(response.read())['access_token']
        else:
            responseCls = BrightboxResponse(response=response.getresponse(), connection=self)
            message = responseCls.parse_error()
            raise InvalidCredsError(message)

    def add_default_headers(self, headers):
        try:
            headers['Authorization'] = 'OAuth ' + self.token
        except AttributeError:
            self.token = self._fetch_oauth_token()
            headers['Authorization'] = 'OAuth ' + self.token
        return headers

    def encode_data(self, data):
        return json.dumps(data)