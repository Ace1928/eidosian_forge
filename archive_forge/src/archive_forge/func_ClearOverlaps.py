from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import argparse
def ClearOverlaps(unused_ref, args, patch_request):
    """Handles clear_overlaps flag."""
    if args.IsSpecified('clear_overlaps'):
        update_mask = patch_request.updateMask
        if not update_mask:
            patch_request.updateMask = 'overlaps'
        else:
            patch_request.updateMask = update_mask + ',' + 'overlaps'
    return patch_request