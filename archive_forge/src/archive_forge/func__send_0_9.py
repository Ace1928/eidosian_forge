import os
import time
from typing import Callable
from . import controldir, errors, osutils, registry, trace
from .branch import Branch
from .i18n import gettext
from .revision import NULL_REVISION
def _send_0_9(branch, revision_id, submit_branch, public_branch, no_patch, no_bundle, message, base_revision_id, local_target_branch=None):
    if not no_bundle:
        if not no_patch:
            patch_type = 'bundle'
        else:
            raise errors.CommandError(gettext('Format 0.9 does not permit bundle with no patch'))
    elif not no_patch:
        patch_type = 'diff'
    else:
        patch_type = None
    from breezy import merge_directive
    return merge_directive.MergeDirective.from_objects(branch.repository, revision_id, time.time(), osutils.local_time_offset(), submit_branch, public_branch=public_branch, patch_type=patch_type, message=message, local_target_branch=local_target_branch)