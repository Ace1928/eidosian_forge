from __future__ import (absolute_import, division, print_function)
from functools import wraps
from os import environ
from os import path
from datetime import datetime
def api_wrapper(func):
    """ Catch API Errors Decorator"""

    @wraps(func)
    def __wrapper(*args, **kwargs):
        module = args[0]
        try:
            return func(*args, **kwargs)
        except core.exceptions.SystemNotFoundException as err:
            module.fail_json(msg=str(err))
        except core.exceptions.APICommandException as err:
            module.fail_json(msg=str(err))
        except Exception as err:
            module.fail_json(msg=str(err))
        return None
    return __wrapper