from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import re
import textwrap
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope.arg_parsers import ArgumentTypeError
from googlecloudsdk.command_lib.util.args import map_util
from googlecloudsdk.core import log
import six
def _ValidateSecrets(secrets_dict):
    """Additional secrets validations that require the entire dict.

  Args:
    secrets_dict: Secrets configuration dict to validate.
  """
    mount_path_to_secret = collections.defaultdict(list)
    for key, value in six.iteritems(secrets_dict):
        if _SECRET_PATH_PATTERN.search(key):
            mount_path = key.split(':')[0]
            secret_res1 = _SECRET_VERSION_SECRET_RESOURCE_PATTERN.search(value).group('secret_resource')
            if mount_path in mount_path_to_secret:
                secret_res_match1 = _SECRET_RESOURCE_PATTERN.search(secret_res1)
                project1 = secret_res_match1.group('project')
                secret1 = secret_res_match1.group('secret')
                for secret_res2 in mount_path_to_secret[mount_path]:
                    secret_res_match2 = _SECRET_RESOURCE_PATTERN.search(secret_res2)
                    project2 = secret_res_match2.group('project')
                    secret2 = secret_res_match2.group('secret')
                    if _SecretsDiffer(project1, secret1, project2, secret2):
                        raise ArgumentTypeError("More than one secret is configured for the mount path '{mount_path}' [violating secrets: {secret1},{secret2}].".format(mount_path=mount_path, secret1=secret1 if project1 == _DEFAULT_PROJECT_IDENTIFIER else secret_res1, secret2=secret2 if project2 == _DEFAULT_PROJECT_IDENTIFIER else secret_res2))
            else:
                mount_path_to_secret[mount_path].append(secret_res1)