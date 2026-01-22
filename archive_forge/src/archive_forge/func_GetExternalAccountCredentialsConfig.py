from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import json
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.credentials import creds as c_creds
from googlecloudsdk.core.credentials import introspect as c_introspect
from googlecloudsdk.core.util import files
def GetExternalAccountCredentialsConfig(filename):
    """Returns the JSON content if the file corresponds to an external account.

  This function is useful when the content of a file need to be inspected first
  before determining how to handle it. More specifically, it would check a
  config file contains an external account cred and return its content which can
  then be used with CredentialsFromAdcDictGoogleAuth (if the contents
  correspond to an external account cred) to avoid having to open the file
  twice.

  Args:
    filename (str): The filepath to the ADC file representing an external
      account credentials.

  Returns:
    Optional(Mapping): The JSON content if the configuration represents an
      external account. Otherwise None is returned.

  Raises:
    BadCredentialFileException: If JSON parsing of the file fails.
  """
    content = files.ReadFileContents(filename)
    try:
        content_json = json.loads(content)
    except ValueError as e:
        raise BadCredentialFileException('Could not read json file {0}: {1}'.format(filename, e))
    if IsExternalAccountConfig(content_json):
        return content_json
    else:
        return None