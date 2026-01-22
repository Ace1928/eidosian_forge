from urllib.parse import urlparse
from urllib.request import url2pathname
from asgiref.sync import sync_to_async
from django.conf import settings
from django.contrib.staticfiles import utils
from django.contrib.staticfiles.views import serve
from django.core.handlers.asgi import ASGIHandler
from django.core.handlers.exception import response_for_exception
from django.core.handlers.wsgi import WSGIHandler, get_path_info
from django.http import Http404
def _should_handle(self, path):
    """
        Check if the path should be handled. Ignore the path if:
        * the host is provided as part of the base_url
        * the request's path isn't under the media path (or equal)
        """
    return path.startswith(self.base_url[2]) and (not self.base_url[1])