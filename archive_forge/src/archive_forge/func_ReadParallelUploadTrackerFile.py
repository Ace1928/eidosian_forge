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
def ReadParallelUploadTrackerFile(tracker_file_name, logger):
    """Read the tracker file from the last parallel composite upload attempt.

  If it exists, the tracker file is of the format described in
  WriteParallelUploadTrackerFile or a legacy format. If the file doesn't exist
  or is formatted incorrectly, then the upload will start from the beginning.

  This function is not thread-safe and must be protected by a lock if
  called within Command.Apply.

  Args:
    tracker_file_name: The name of the tracker file to read parse.
    logger: logging.Logger for outputting log messages.

  Returns:
    enc_key_sha256: Encryption key SHA256 used to encrypt the existing
        components, or None if an encryption key was not used.
    component_prefix: String prefix used in naming the existing components, or
        None if no prefix was found.
    existing_components: A list of ObjectFromTracker objects representing
        the set of files that have already been uploaded.
  """
    enc_key_sha256 = None
    prefix = None
    existing_components = []
    tracker_file = None
    try:
        tracker_file = open(tracker_file_name, 'r')
        tracker_data = tracker_file.read()
        tracker_json = json.loads(tracker_data)
        enc_key_sha256 = tracker_json[_CompositeUploadTrackerEntry.ENC_SHA256]
        prefix = tracker_json[_CompositeUploadTrackerEntry.PREFIX]
        for component in tracker_json[_CompositeUploadTrackerEntry.COMPONENTS_LIST]:
            existing_components.append(ObjectFromTracker(component[_CompositeUploadTrackerEntry.COMPONENT_NAME], component[_CompositeUploadTrackerEntry.COMPONENT_GENERATION]))
    except IOError as e:
        if e.errno != errno.ENOENT:
            logger.warn("Couldn't read upload tracker file (%s): %s. Restarting parallel composite upload from scratch.", tracker_file_name, e.strerror)
    except (KeyError, ValueError) as e:
        enc_key_sha256 = None
        prefix, existing_components = _ParseLegacyTrackerData(tracker_data)
    finally:
        if tracker_file:
            tracker_file.close()
    return (enc_key_sha256, prefix, existing_components)