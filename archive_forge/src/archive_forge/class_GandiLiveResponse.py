import json
from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.types import ProviderError
class GandiLiveResponse(JsonResponse):
    """
    A Base Gandi Live Response class to derive from.
    """

    def success(self):
        """
        Determine if our request was successful.

        For the Gandi Live response class, tag all responses as successful and
        raise appropriate Exceptions from parse_body.

        :return: C{True}
        """
        return True

    def parse_body(self):
        """
        Parse the JSON response body, or raise exceptions as appropriate.

        :return:  JSON dictionary
        :rtype:   ``dict``
        """
        json_error = False
        try:
            body = json.loads(self.body)
        except Exception:
            body = self.body
            json_error = True
        valid_http_codes = [httplib.OK, httplib.CREATED]
        if self.status in valid_http_codes:
            if json_error:
                raise JsonParseError(body, self.status)
            else:
                return body
        elif self.status == httplib.NO_CONTENT:
            if len(body) > 0:
                msg = '"No Content" response contained content'
                raise GandiLiveBaseError(msg, self.status)
            else:
                return {}
        elif self.status == httplib.NOT_FOUND:
            message = self._get_error(body, json_error)
            raise ResourceNotFoundError(message, self.status)
        elif self.status == httplib.BAD_REQUEST:
            message = self._get_error(body, json_error)
            raise InvalidRequestError(message, self.status)
        elif self.status == httplib.CONFLICT:
            message = self._get_error(body, json_error)
            raise ResourceConflictError(message, self.status)
        else:
            message = self._get_error(body, json_error)
            raise GandiLiveBaseError(message, self.status)

    def _get_error(self, body, json_error):
        """
        Get the error code and message from a JSON response.

        Incorporate the first error if there are multiple errors.

        :param  body: The body of the JSON response dictionary
        :type   body: ``dict``

        :return:  String containing error message
        :rtype:   ``str``
        """
        if not json_error and 'cause' in body:
            message = '{}: {}'.format(body['cause'], body['message'])
            if 'errors' in body:
                err = body['errors'][0]
                message = '{} ({} in {}: {})'.format(message, err.get('location'), err.get('name'), err.get('description'))
        else:
            message = body
        return message