import base64
from libcloud.utils.py3 import b
from libcloud.common.base import JsonResponse, ConnectionUserAndKey
from libcloud.common.types import ProviderError
class LiquidWebResponse(JsonResponse):
    objects = None
    errors = None

    def __init__(self, response, connection):
        self.errors = []
        super().__init__(response=response, connection=connection)
        self.objects, self.errors = self.parse_body_and_errors()
        if self.errors:
            error = self.errors.pop()
            raise self._make_excp(error, self.status)

    def parse_body_and_errors(self):
        data = []
        errors = []
        js = super().parse_body()
        if 'items' in js:
            data.append(js['items'])
        if 'name' in js:
            data.append(js)
        if 'deleted' in js:
            data.append(js['deleted'])
        if 'error_class' in js:
            errors.append(js)
        return (data, errors)

    def success(self):
        """
        Returns ``True`` if our request is successful.
        """
        return len(self.errors) == 0

    def _make_excp(self, error, status):
        """
        Raise LiquidWebException.
        """
        exc_type = error.get('error_class')
        message = error.get('full_message')
        try:
            _type = EXCEPTIONS_FIELDS[exc_type]
            fields = _type.get('fields')
            extra = {}
        except KeyError:
            fields = []
        for field in fields:
            extra[field] = error.get(field)
        return APIException(exc_type, message, status, extra=extra)