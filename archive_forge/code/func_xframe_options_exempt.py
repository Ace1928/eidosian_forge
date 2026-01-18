from functools import wraps
from asgiref.sync import iscoroutinefunction
def xframe_options_exempt(view_func):
    """
    Modify a view function by setting a response variable that instructs
    XFrameOptionsMiddleware to NOT set the X-Frame-Options HTTP header. Usage:

    @xframe_options_exempt
    def some_view(request):
        ...
    """
    if iscoroutinefunction(view_func):

        async def _view_wrapper(*args, **kwargs):
            response = await view_func(*args, **kwargs)
            response.xframe_options_exempt = True
            return response
    else:

        def _view_wrapper(*args, **kwargs):
            response = view_func(*args, **kwargs)
            response.xframe_options_exempt = True
            return response
    return wraps(view_func)(_view_wrapper)