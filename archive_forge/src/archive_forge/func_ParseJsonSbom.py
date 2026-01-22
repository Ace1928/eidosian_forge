from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
import json
import random
import re
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from containerregistry.client import docker_creds
from containerregistry.client import docker_name
from containerregistry.client.v2_2 import docker_http as v2_2_docker_http
from containerregistry.client.v2_2 import docker_image as v2_2_image
from containerregistry.client.v2_2 import docker_image_list as v2_2_image_list
from googlecloudsdk.api_lib.artifacts import exceptions as ar_exceptions
from googlecloudsdk.api_lib.cloudkms import base as cloudkms_base
from googlecloudsdk.api_lib.container.images import util as gcr_util
from googlecloudsdk.api_lib.containeranalysis import filter_util
from googlecloudsdk.api_lib.containeranalysis import requests as ca_requests
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.command_lib.artifacts import docker_util
from googlecloudsdk.command_lib.artifacts import requests as ar_requests
from googlecloudsdk.command_lib.artifacts import util
from googlecloudsdk.command_lib.projects import util as project_util
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core import transports
from googlecloudsdk.core.util import files
import requests
import six
from six.moves import urllib
def ParseJsonSbom(file_path):
    """Retrieves information about a docker image based on the fully-qualified name.

  Args:
    file_path: str, The sbom file location.

  Raises:
    ar_exceptions.InvalidInputValueError: If the sbom format is not supported.

  Returns:
    An SbomFile object with metadata of the given sbom.
  """
    try:
        content = files.ReadFileContents(file_path)
        data = json.loads(content)
    except ValueError as e:
        raise ar_exceptions.InvalidInputValueError('The file is not a valid JSON file', e)
    except files.Error as e:
        raise ar_exceptions.InvalidInputValueError('Failed to read the sbom file', e)
    if 'spdxVersion' in data:
        res = _ParseSpdx(data)
    elif data.get('bomFormat') == 'CycloneDX':
        res = _ParseCycloneDx(data)
    else:
        raise ar_exceptions.InvalidInputValueError(_UNSUPPORTED_SBOM_FORMAT_ERROR)
    sha256_digest = hashlib.sha256(six.ensure_binary(content)).hexdigest()
    res.digests['sha256'] = sha256_digest
    return res