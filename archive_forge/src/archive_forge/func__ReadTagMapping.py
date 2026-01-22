from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import itertools
import json
import re
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.operations import poller
from googlecloudsdk.api_lib.resource_manager import tags as rm_tags
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.network_firewall_policies import convert_terraform
from googlecloudsdk.command_lib.compute.network_firewall_policies import secure_tags_utils
from googlecloudsdk.command_lib.compute.networks import flags as network_flags
from googlecloudsdk.command_lib.resource_manager import endpoint_utils as endpoints
from googlecloudsdk.command_lib.resource_manager import operations
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.util import files
def _ReadTagMapping(file_name):
    """Imports legacy to secure tag mapping from a JSON file."""
    try:
        with files.FileReader(file_name) as f:
            data = json.load(f)
    except FileNotFoundError:
        log.status.Print("File '{file}' was not found. Tag mapping was not imported.".format(file=file_name))
        return None
    except OSError:
        log.status.Print("OS error occurred when opening the file '{file}'. Tag mapping was not imported.".format(file=file_name))
        return None
    except Exception as e:
        log.status.Print("Unexpected error occurred when reading the JSON file '{file}'. Tag mapping was not imported.".format(file=file_name))
        log.status.Print(repr(e))
        return None
    tag_mapping = {k: secure_tags_utils.TranslateSecureTag(v) for k, v in data.items()}
    return tag_mapping