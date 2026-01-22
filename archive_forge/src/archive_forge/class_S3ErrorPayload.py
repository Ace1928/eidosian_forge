from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.api_lib.util import resource
from googlecloudsdk.core import exceptions as core_exceptions
from six.moves import urllib
class S3ErrorPayload(api_exceptions.FormattableErrorPayload):
    """Allows using format strings to create strings from botocore ClientErrors.

  Format strings of the form '{field_name}' will be populated from class
  attributes. Strings of the form '{.field_name}' will be populated from the
  self.content JSON dump. See api_lib.util.HttpErrorPayload for more detail and
  sample usage.

  Attributes:
    content (dict): The dumped JSON content.
    message (str): The human readable error message.
    status_code (int): The HTTP status code number.
    status_description (str): The status_code description.
    status_message (str): Context specific status message.
  """

    def __init__(self, client_error):
        """Initializes an S3ErrorPayload instance.

    Args:
      client_error (Union[botocore.exceptions.ClientError, str]): The error
        thrown by botocore, or a string that will be displayed as the error
        message.
    """
        super(S3ErrorPayload, self).__init__(client_error)
        self.botocore_error_string = str(client_error)
        if hasattr(client_error, 'response'):
            self.content = client_error.response
            if 'ResponseMetadata' in client_error.response:
                self.status_code = client_error.response['ResponseMetadata'].get('HttpStatusCode', 0)
            if 'Error' in client_error.response:
                error = client_error.response['Error']
                self.status_description = error.get('Code', '')
                self.status_message = error.get('Message', '')
            self.message = self._MakeGenericMessage()