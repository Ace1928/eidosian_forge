from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.middleware.csrf import CsrfViewMiddleware, get_token
from django.utils.decorators import decorator_from_middleware
class _EnsureCsrfToken(CsrfViewMiddleware):

    def _reject(self, request, reason):
        return None