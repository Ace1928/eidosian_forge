from oslo_utils import reflection
import webob.exc
from heat.common.i18n import _
from heat.common import serializers
class HeatAPIException(webob.exc.HTTPError):
    """webob HTTPError subclass that creates a serialized body.

    Subclass webob HTTPError so we can correctly serialize the wsgi response
    into the http response body, using the format specified by the request.
    Note this should not be used directly, instead use the subclasses
    defined below which map to AWS API errors.
    """
    code = 400
    title = 'HeatAPIException'
    explanation = _('Generic HeatAPIException, please use specific subclasses!')
    err_type = 'Sender'

    def __init__(self, detail=None):
        """Overload HTTPError constructor to create a default serialized body.

        This is required because not all error responses are processed
        by the wsgi controller (such as auth errors), which are further up the
        paste pipeline.  We serialize in XML by default (as AWS does).
        """
        webob.exc.HTTPError.__init__(self, detail=detail)
        serializer = serializers.XMLResponseSerializer()
        serializer.default(self, self.get_unserialized_body())

    def get_unserialized_body(self):
        """Return a dict suitable for serialization in the wsgi controller.

        This wraps the exception details in a format which maps to the
        expected format for the AWS API.
        """
        if self.detail:
            message = ':'.join([self.explanation, self.detail])
        else:
            message = self.explanation
        return {'ErrorResponse': {'Error': {'Type': self.err_type, 'Code': self.title, 'Message': message}}}