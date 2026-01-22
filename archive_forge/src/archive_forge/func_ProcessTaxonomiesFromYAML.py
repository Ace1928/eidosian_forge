from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.data_catalog import util as api_util
from googlecloudsdk.core import exceptions
import six
def ProcessTaxonomiesFromYAML(inline_source, version_label):
    """Converts the given inline source dict to the corresponding import request.

  Args:
    inline_source: dict, inlineSource part of the import taxonomy request.
    version_label: string, specifies the version for client.
  Returns:
    GoogleCloudDatacatalogV1beta1ImportTaxonomiesRequest
  Raises:
    InvalidInlineSourceError: If the inline source is invalid.
  """
    messages = api_util.GetMessagesModule(version_label)
    if version_label == 'v1':
        request = messages.GoogleCloudDatacatalogV1ImportTaxonomiesRequest
    else:
        request = messages.GoogleCloudDatacatalogV1beta1ImportTaxonomiesRequest
    try:
        import_request_message = encoding.DictToMessage({'inlineSource': inline_source}, request)
    except AttributeError:
        raise InvalidInlineSourceError('An error occurred while parsing the serialized taxonomy. Please check your input file.')
    unrecognized_field_paths = _GetUnrecognizedFieldPaths(import_request_message)
    if unrecognized_field_paths:
        error_msg_lines = ['Invalid inline source, the following fields are ' + 'unrecognized:']
        error_msg_lines += unrecognized_field_paths
        raise InvalidInlineSourceError('\n'.join(error_msg_lines))
    return import_request_message