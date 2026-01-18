from functools import wraps
from libcloud.common.types import LibcloudError
def wrap_non_libcloud_exceptions(func):
    """
    Decorators function which catches non LibcloudError exceptions, wraps them
    in LibcloudError class and re-throws the wrapped exception.

    Note: This function should only be used to wrap methods on the driver
    classes.
    """

    @wraps(func)
    def decorated_function(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if isinstance(e, LibcloudError):
                raise e
            if len(args) >= 1:
                driver = args[0]
            else:
                driver = None
            fault = getattr(e, 'fault', None)
            if fault and getattr(fault, 'string', None):
                message = fault.string
            else:
                message = str(e)
            raise LibcloudError(value=message, driver=driver)
    return decorated_function