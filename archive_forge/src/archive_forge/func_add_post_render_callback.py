from django.http import HttpResponse
from .loader import get_template, select_template
def add_post_render_callback(self, callback):
    """Add a new post-rendering callback.

        If the response has already been rendered,
        invoke the callback immediately.
        """
    if self._is_rendered:
        callback(self)
    else:
        self._post_render_callbacks.append(callback)