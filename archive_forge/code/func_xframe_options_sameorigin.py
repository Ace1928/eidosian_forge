from functools import wraps
from asgiref.sync import iscoroutinefunction
def xframe_options_sameorigin(view_func):
    """
    Modify a view function so its response has the X-Frame-Options HTTP
    header set to 'SAMEORIGIN' as long as the response doesn't already have
    that header set. Usage:

    @xframe_options_sameorigin
    def some_view(request):
        ...
    """
    if iscoroutinefunction(view_func):

        async def _view_wrapper(*args, **kwargs):
            response = await view_func(*args, **kwargs)
            if response.get('X-Frame-Options') is None:
                response['X-Frame-Options'] = 'SAMEORIGIN'
            return response
    else:

        def _view_wrapper(*args, **kwargs):
            response = view_func(*args, **kwargs)
            if response.get('X-Frame-Options') is None:
                response['X-Frame-Options'] = 'SAMEORIGIN'
            return response
    return wraps(view_func)(_view_wrapper)