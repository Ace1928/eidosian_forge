from libcloud.utils.py3 import httplib
from libcloud.common.base import JsonResponse, ConnectionKey
from libcloud.common.gandi import BaseObject
from libcloud.common.types import LibcloudError, InvalidCredsError
class LinodeResponseV4(JsonResponse):
    valid_response_codes = [httplib.OK, httplib.NO_CONTENT]

    def parse_body(self):
        """Parse the body of the response into JSON objects
        :return: ``dict`` of objects"""
        return super().parse_body()

    def parse_error(self):
        """
        Parse the error body and raise the appropriate exception
        """
        status = int(self.status)
        data = self.parse_body()
        error = data['errors'][0]
        reason = error.get('reason')
        field = error.get('field')
        if field is not None:
            error_msg = '{}-{}'.format(reason, field)
        else:
            error_msg = reason
        if status in [httplib.UNAUTHORIZED, httplib.FORBIDDEN]:
            raise InvalidCredsError(value=error_msg)
        raise LibcloudError('%s Status code: %d.' % (error_msg, status), driver=self.connection.driver)

    def success(self):
        """Check the response for success
        :return: ``bool`` indicating a successful request"""
        return self.status in self.valid_response_codes