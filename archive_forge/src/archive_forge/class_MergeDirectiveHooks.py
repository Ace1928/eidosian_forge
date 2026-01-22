import base64
import contextlib
import re
from io import BytesIO
from . import lazy_import
from breezy import (
from breezy.bzr import (
from breezy.bzr.bundle import (
from . import errors, hooks, registry
class MergeDirectiveHooks(hooks.Hooks):
    """Hooks for MergeDirective classes."""

    def __init__(self):
        hooks.Hooks.__init__(self, 'breezy.merge_directive', 'BaseMergeDirective.hooks')
        self.add_hook('merge_request_body', 'Called with a MergeRequestBodyParams when a body is needed for a merge request.  Callbacks must return a body.  If more than one callback is registered, the output of one callback is provided to the next.', (1, 15, 0))