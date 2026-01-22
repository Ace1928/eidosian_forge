from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import enum
import os.path
import string
import uuid
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import daisy_utils
from googlecloudsdk.api_lib.compute import image_utils
from googlecloudsdk.api_lib.compute import utils
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute.images import flags
from googlecloudsdk.command_lib.compute.images import os_choices
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import progress_tracker
import six
def _CheckForExistingImage(image_name, compute_holder, arg_name='IMAGE_NAME', expect_to_exist=False):
    """Check if image already exists."""
    expect_to_exist_image_name_exclusions = ['sample-image-123']
    if expect_to_exist and image_name in expect_to_exist_image_name_exclusions:
        return
    image_ref = resources.REGISTRY.Parse(image_name, collection='compute.images', params={'project': properties.VALUES.core.project.GetOrFail})
    image_expander = image_utils.ImageExpander(compute_holder.client, compute_holder.resources)
    try:
        _ = image_expander.GetImage(image_ref)
        image_exists = True
    except utils.ImageNotFoundError:
        image_exists = False
    if not expect_to_exist and image_exists:
        message = 'The image [{0}] already exists.'.format(image_name)
        raise exceptions.InvalidArgumentException(arg_name, message)
    elif expect_to_exist and (not image_exists):
        message = 'The image [{0}] does not exist.'.format(image_name)
        raise exceptions.InvalidArgumentException(arg_name, message)