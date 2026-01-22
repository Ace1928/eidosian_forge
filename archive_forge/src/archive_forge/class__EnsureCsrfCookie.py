from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.middleware.csrf import CsrfViewMiddleware, get_token
from django.utils.decorators import decorator_from_middleware
class _EnsureCsrfCookie(CsrfViewMiddleware):

    def _reject(self, request, reason):
        return None

    def process_view(self, request, callback, callback_args, callback_kwargs):
        retval = super().process_view(request, callback, callback_args, callback_kwargs)
        get_token(request)
        return retval