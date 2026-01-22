from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from collections import namedtuple
import errno
import json
import random
import six
import gslib
from gslib.exception import CommandException
from gslib.tracker_file import (WriteJsonDataToTrackerFile,
from gslib.utils.constants import UTF8
def ValidateParallelCompositeTrackerData(tracker_file_name, existing_enc_sha256, existing_prefix, existing_components, current_enc_key_sha256, bucket_url, command_obj, logger, delete_func, delete_exc_handler):
    """Validates that tracker data matches the current encryption key.

  If the data does not match, makes a best-effort attempt to delete existing
  temporary component objects encrypted with the old key.

  Args:
    tracker_file_name: String file name of tracker file.
    existing_enc_sha256: Encryption key SHA256 used to encrypt the existing
        components, or None if an encryption key was not used.
    existing_prefix: String prefix used in naming the existing components, or
        None if no prefix was found.
    existing_components: A list of ObjectFromTracker objects representing
        the set of files that have already been uploaded.
    current_enc_key_sha256: Current Encryption key SHA256 that should be used
        to encrypt objects.
    bucket_url: Bucket URL in which the components exist.
    command_obj: Command class for calls to Apply.
    logger: logging.Logger for outputting log messages.
    delete_func: command.Apply-callable function for deleting objects.
    delete_exc_handler: Exception handler for delete_func.

  Returns:
    prefix: existing_prefix, or None if the encryption key did not match.
    existing_components: existing_components, or empty list if the encryption
        key did not match.
  """
    if six.PY3:
        if isinstance(existing_enc_sha256, str):
            existing_enc_sha256 = existing_enc_sha256.encode(UTF8)
        if isinstance(current_enc_key_sha256, str):
            current_enc_key_sha256 = current_enc_key_sha256.encode(UTF8)
    if existing_prefix and existing_enc_sha256 != current_enc_key_sha256:
        try:
            logger.warn('Upload tracker file (%s) does not match current encryption key. Deleting old components and restarting upload from scratch with a new tracker file that uses the current encryption key.', tracker_file_name)
            components_to_delete = []
            for component in existing_components:
                url = bucket_url.Clone()
                url.object_name = component.object_name
                url.generation = component.generation
            command_obj.Apply(delete_func, components_to_delete, delete_exc_handler, arg_checker=gslib.command.DummyArgChecker, parallel_operations_override=command_obj.ParallelOverrideReason.SPEED)
        except:
            component_names = [component.object_name for component in existing_components]
            logger.warn('Failed to delete some of the following temporary objects:\n%s\n(Continuing on to re-upload components from scratch.)', '\n'.join(component_names))
        return (None, [])
    return (existing_prefix, existing_components)