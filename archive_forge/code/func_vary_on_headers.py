from functools import wraps
from asgiref.sync import iscoroutinefunction
from django.utils.cache import patch_vary_headers
def vary_on_headers(*headers):
    """
    A view decorator that adds the specified headers to the Vary header of the
    response. Usage:

       @vary_on_headers('Cookie', 'Accept-language')
       def index(request):
           ...

    Note that the header names are not case-sensitive.
    """

    def decorator(func):
        if iscoroutinefunction(func):

            async def _view_wrapper(request, *args, **kwargs):
                response = await func(request, *args, **kwargs)
                patch_vary_headers(response, headers)
                return response
        else:

            def _view_wrapper(request, *args, **kwargs):
                response = func(request, *args, **kwargs)
                patch_vary_headers(response, headers)
                return response
        return wraps(func)(_view_wrapper)
    return decorator