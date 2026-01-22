import re
from . import errors, osutils, transport
class DisabledViews(_Views):
    """View storage that refuses to store anything.

    This is used by older formats that can't store views.
    """

    def __init__(self, tree):
        self.tree = tree

    def supports_views(self):
        return False

    def _not_supported(self, *a, **k):
        raise ViewsNotSupported(self.tree)
    get_view_info = _not_supported
    set_view_info = _not_supported
    lookup_view = _not_supported
    set_view = _not_supported
    delete_view = _not_supported