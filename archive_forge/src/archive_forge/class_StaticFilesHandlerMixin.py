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
class StaticFilesHandlerMixin:
    """
    Common methods used by WSGI and ASGI handlers.
    """
    handles_files = True

    def load_middleware(self):
        pass

    def get_base_url(self):
        utils.check_settings()
        return settings.STATIC_URL

    def _should_handle(self, path):
        """
        Check if the path should be handled. Ignore the path if:
        * the host is provided as part of the base_url
        * the request's path isn't under the media path (or equal)
        """
        return path.startswith(self.base_url[2]) and (not self.base_url[1])

    def file_path(self, url):
        """
        Return the relative path to the media file on disk for the given URL.
        """
        relative_url = url.removeprefix(self.base_url[2])
        return url2pathname(relative_url)

    def serve(self, request):
        """Serve the request path."""
        return serve(request, self.file_path(request.path), insecure=True)

    def get_response(self, request):
        try:
            return self.serve(request)
        except Http404 as e:
            return response_for_exception(request, e)

    async def get_response_async(self, request):
        try:
            return await sync_to_async(self.serve, thread_sensitive=False)(request)
        except Http404 as e:
            return await sync_to_async(response_for_exception, thread_sensitive=False)(request, e)