from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.app import appengine_api_client
from googlecloudsdk.api_lib.app import operations_util
from googlecloudsdk.api_lib.cloudresourcemanager import projects_api
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.iam import iam_util
from googlecloudsdk.command_lib.projects import util as projects_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
import six
def _ParseIapSettingsFile(self, iap_settings_file_path, iap_settings_message_type):
    """Create an iap settings message from a JSON formatted file.

    Args:
       iap_settings_file_path: Path to iap_setttings JSON file
       iap_settings_message_type: iap settings message type to convert JSON to

    Returns:
       the iap_settings message filled from JSON file
    Raises:
       BadFileException if JSON file is malformed.
    """
    iap_settings_to_parse = yaml.load_path(iap_settings_file_path)
    try:
        iap_settings_message = encoding.PyValueToMessage(iap_settings_message_type, iap_settings_to_parse)
    except AttributeError as e:
        raise calliope_exceptions.BadFileException('Iap settings file {0} does not contain properly formatted JSON {1}'.format(iap_settings_file_path, six.text_type(e)))
    return iap_settings_message