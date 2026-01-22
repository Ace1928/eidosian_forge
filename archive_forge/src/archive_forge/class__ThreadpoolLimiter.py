import os
import re
import sys
import ctypes
import textwrap
from typing import final
import warnings
from ctypes.util import find_library
from abc import ABC, abstractmethod
from functools import lru_cache
from contextlib import ContextDecorator
class _ThreadpoolLimiter:
    """The guts of ThreadpoolController.limit

    Refer to the docstring of ThreadpoolController.limit for more details.

    It will only act on the library controllers held by the provided `controller`.
    Using the default constructor sets the limits right away such that it can be used as
    a callable. Setting the limits can be delayed by using the `wrap` class method such
    that it can be used as a decorator.
    """

    def __init__(self, controller, *, limits=None, user_api=None):
        self._controller = controller
        self._limits, self._user_api, self._prefixes = self._check_params(limits, user_api)
        self._original_info = self._controller.info()
        self._set_threadpool_limits()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.restore_original_limits()

    @classmethod
    def wrap(cls, controller, *, limits=None, user_api=None):
        """Return an instance of this class that can be used as a decorator"""
        return _ThreadpoolLimiterDecorator(controller=controller, limits=limits, user_api=user_api)

    def restore_original_limits(self):
        """Set the limits back to their original values"""
        for lib_controller, original_info in zip(self._controller.lib_controllers, self._original_info):
            lib_controller.set_num_threads(original_info['num_threads'])
    unregister = restore_original_limits

    def get_original_num_threads(self):
        """Original num_threads from before calling threadpool_limits

        Return a dict `{user_api: num_threads}`.
        """
        num_threads = {}
        warning_apis = []
        for user_api in self._user_api:
            limits = [lib_info['num_threads'] for lib_info in self._original_info if lib_info['user_api'] == user_api]
            limits = set(limits)
            n_limits = len(limits)
            if n_limits == 1:
                limit = limits.pop()
            elif n_limits == 0:
                limit = None
            else:
                limit = min(limits)
                warning_apis.append(user_api)
            num_threads[user_api] = limit
        if warning_apis:
            warnings.warn('Multiple value possible for following user apis: ' + ', '.join(warning_apis) + '. Returning the minimum.')
        return num_threads

    def _check_params(self, limits, user_api):
        """Suitable values for the _limits, _user_api and _prefixes attributes"""
        if isinstance(limits, str) and limits == 'sequential_blas_under_openmp':
            limits, user_api = self._controller._get_params_for_sequential_blas_under_openmp().values()
        if limits is None or isinstance(limits, int):
            if user_api is None:
                user_api = _ALL_USER_APIS
            elif user_api in _ALL_USER_APIS:
                user_api = [user_api]
            else:
                raise ValueError(f'user_api must be either in {_ALL_USER_APIS} or None. Got {user_api} instead.')
            if limits is not None:
                limits = {api: limits for api in user_api}
            prefixes = []
        else:
            if isinstance(limits, list):
                limits = {lib_info['prefix']: lib_info['num_threads'] for lib_info in limits}
            elif isinstance(limits, ThreadpoolController):
                limits = {lib_controller.prefix: lib_controller.num_threads for lib_controller in limits.lib_controllers}
            if not isinstance(limits, dict):
                raise TypeError(f"limits must either be an int, a list, a dict, or 'sequential_blas_under_openmp'. Got {type(limits)} instead")
            prefixes = [prefix for prefix in limits if prefix in _ALL_PREFIXES]
            user_api = [api for api in limits if api in _ALL_USER_APIS]
        return (limits, user_api, prefixes)

    def _set_threadpool_limits(self):
        """Change the maximal number of threads in selected thread pools.

        Return a list with all the supported libraries that have been found
        matching `self._prefixes` and `self._user_api`.
        """
        if self._limits is None:
            return
        for lib_controller in self._controller.lib_controllers:
            if lib_controller.prefix in self._limits:
                num_threads = self._limits[lib_controller.prefix]
            elif lib_controller.user_api in self._limits:
                num_threads = self._limits[lib_controller.user_api]
            else:
                continue
            if num_threads is not None:
                lib_controller.set_num_threads(num_threads)