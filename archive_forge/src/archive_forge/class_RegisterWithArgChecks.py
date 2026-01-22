import logging
import traceback
from os_ken.services.protocols.bgp.base import add_bgp_error_metadata
from os_ken.services.protocols.bgp.base import API_ERROR_CODE
from os_ken.services.protocols.bgp.base import BGPSException
from os_ken.services.protocols.bgp.core_manager import CORE_MANAGER
from os_ken.services.protocols.bgp.rtconf.base import get_validator
from os_ken.services.protocols.bgp.rtconf.base import MissingRequiredConf
from os_ken.services.protocols.bgp.rtconf.base import RuntimeConfigError
class RegisterWithArgChecks(object):
    """Decorator for registering API functions.

    Does some argument checking and validation of required arguments.
    """

    def __init__(self, name, req_args=None, opt_args=None):
        self._name = name
        if not req_args:
            req_args = []
        self._req_args = req_args
        if not opt_args:
            opt_args = []
        self._opt_args = opt_args
        self._all_args = set(self._req_args) | set(self._opt_args)

    def __call__(self, func):
        """Wraps given function and registers it as API.

            Returns original function.
        """

        def wrapped_fun(**kwargs):
            """Wraps a function to do validation before calling actual func.

            Wraps a function to take key-value args. only. Checks if:
            1) all required argument of wrapped function are provided
            2) no extra/un-known arguments are passed
            3) checks if validator for required arguments is available
            4) validates required arguments
            5) if validator for optional arguments is registered,
               validates optional arguments.
            Raises exception if no validator can be found for required args.
            """
            if not kwargs and len(self._req_args) > 0:
                raise MissingRequiredConf(desc='Missing all required attributes.')
            given_args = set(kwargs.keys())
            unknown_attrs = given_args - set(self._all_args)
            if unknown_attrs:
                raise RuntimeConfigError(desc='Unknown attributes %r' % unknown_attrs)
            missing_req_args = set(self._req_args) - given_args
            if missing_req_args:
                conf_name = ', '.join(missing_req_args)
                raise MissingRequiredConf(conf_name=conf_name)
            req_values = []
            for req_arg in self._req_args:
                req_value = kwargs.get(req_arg)
                validator = get_validator(req_arg)
                if not validator:
                    raise ValueError('No validator registered for function=%s and arg=%s' % (func, req_arg))
                validator(req_value)
                req_values.append(req_value)
            opt_items = {}
            for opt_arg, opt_value in kwargs.items():
                if opt_arg in self._opt_args:
                    validator = get_validator(opt_arg)
                    if validator:
                        validator(opt_value)
                    opt_items[opt_arg] = opt_value
            return func(*req_values, **opt_items)
        _CALL_REGISTRY[self._name] = wrapped_fun
        return func