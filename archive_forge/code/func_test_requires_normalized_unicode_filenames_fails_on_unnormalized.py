import os
import sys
from io import StringIO
from ... import add as _mod_add
from ... import errors, ignores, osutils, tests, trace, transport, workingtree
from .. import features, per_workingtree, test_smart_add
def test_requires_normalized_unicode_filenames_fails_on_unnormalized(self):
    """Adding unnormalized unicode filenames fail if and only if the
        workingtree format has the requires_normalized_unicode_filenames flag
        set and the underlying filesystem doesn't normalize.
        """
    osutils.normalized_filename = osutils._accessible_normalized_filename
    if self.workingtree_format.requires_normalized_unicode_filenames and sys.platform != 'darwin':
        self.assertRaises(transport.NoSuchFile, self.wt.smart_add, ['å'])
    else:
        self.wt.smart_add(['å'])