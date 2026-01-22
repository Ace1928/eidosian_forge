from __future__ import with_statement
import logging
import os
from textwrap import dedent
import time
import pathlib
import pydoc
import re
import functools
import traceback
import warnings
import inspect
import weakref
from datetime import timedelta
from tokenize import open as open_py_source
from . import hashing
from .func_inspect import get_func_code, get_func_name, filter_args
from .func_inspect import format_call
from .func_inspect import format_signature
from .logger import Logger, format_time, pformat
from ._store_backends import StoreBackendBase, FileSystemStoreBackend
from ._store_backends import CacheWarning  # noqa
class MemorizedFunc(Logger):
    """Callable object decorating a function for caching its return value
    each time it is called.

    Methods are provided to inspect the cache or clean it.

    Attributes
    ----------
    func: callable
        The original, undecorated, function.

    location: string
        The location of joblib cache. Depends on the store backend used.

    backend: str
        Type of store backend for reading/writing cache files.
        Default is 'local', in which case the location is the path to a
        disk storage.

    ignore: list or None
        List of variable names to ignore when choosing whether to
        recompute.

    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}
        The memmapping mode used when loading from cache
        numpy arrays. See numpy.load for the meaning of the different
        values.

    compress: boolean, or integer
        Whether to zip the stored data on disk. If an integer is
        given, it should be between 1 and 9, and sets the amount
        of compression. Note that compressed arrays cannot be
        read by memmapping.

    verbose: int, optional
        The verbosity flag, controls messages that are issued as
        the function is evaluated.

    cache_validation_callback: callable, optional
        Callable to check if a result in cache is valid or is to be recomputed.
        When the function is called with arguments for which a cache exists,
        the callback is called with the cache entry's metadata as its sole
        argument. If it returns True, the cached result is returned, else the
        cache for these arguments is cleared and the result is recomputed.
    """

    def __init__(self, func, location, backend='local', ignore=None, mmap_mode=None, compress=False, verbose=1, timestamp=None, cache_validation_callback=None):
        Logger.__init__(self)
        self.mmap_mode = mmap_mode
        self.compress = compress
        self.func = func
        self.cache_validation_callback = cache_validation_callback
        if ignore is None:
            ignore = []
        self.ignore = ignore
        self._verbose = verbose
        self.store_backend = _store_backend_factory(backend, location, verbose=verbose, backend_options=dict(compress=compress, mmap_mode=mmap_mode))
        if self.store_backend is not None:
            self.store_backend.store_cached_func_code([_build_func_identifier(self.func)])
        if timestamp is None:
            timestamp = time.time()
        self.timestamp = timestamp
        try:
            functools.update_wrapper(self, func)
        except Exception:
            " Objects like ufunc don't like that "
        if inspect.isfunction(func):
            doc = pydoc.TextDoc().document(func)
            doc = doc.replace('\n', '\n\n', 1)
            doc = re.sub('\x08.', '', doc)
        else:
            doc = func.__doc__
        self.__doc__ = 'Memoized version of %s' % doc
        self._func_code_info = None
        self._func_code_id = None

    def _is_in_cache_and_valid(self, path):
        """Check if the function call is cached and valid for given arguments.

        - Compare the function code with the one from the cached function,
        asserting if it has changed.
        - Check if the function call is present in the cache.
        - Call `cache_validation_callback` for user define cache validation.

        Returns True if the function call is in cache and can be used, and
        returns False otherwise.
        """
        if not self._check_previous_func_code(stacklevel=4):
            return False
        if not self.store_backend.contains_item(path):
            return False
        metadata = self.store_backend.get_metadata(path)
        if self.cache_validation_callback is not None and (not self.cache_validation_callback(metadata)):
            self.store_backend.clear_item(path)
            return False
        return True

    def _cached_call(self, args, kwargs, shelving=False):
        """Call wrapped function and cache result, or read cache if available.

        This function returns the wrapped function output and some metadata.

        Arguments:
        ----------

        args, kwargs: list and dict
            input arguments for wrapped function

        shelving: bool
            True when called via the call_and_shelve function.


        Returns
        -------
        output: value or tuple or None
            Output of the wrapped function.
            If shelving is True and the call has been already cached,
            output is None.

        argument_hash: string
            Hash of function arguments.

        metadata: dict
            Some metadata about wrapped function call (see _persist_input()).
        """
        func_id, args_id = self._get_output_identifiers(*args, **kwargs)
        metadata = None
        msg = None
        must_call = False
        if self._verbose >= 20:
            logging.basicConfig(level=logging.INFO)
            _, name = get_func_name(self.func)
            location = self.store_backend.get_cached_func_info([func_id])['location']
            _, signature = format_signature(self.func, *args, **kwargs)
            self.info(dedent(f'\n                        Querying {name} with signature\n                        {signature}.\n\n                        (argument hash {args_id})\n\n                        The store location is {location}.\n                        '))
        if self._is_in_cache_and_valid([func_id, args_id]):
            try:
                t0 = time.time()
                if self._verbose:
                    msg = _format_load_msg(func_id, args_id, timestamp=self.timestamp, metadata=metadata)
                if not shelving:
                    out = self.store_backend.load_item([func_id, args_id], msg=msg, verbose=self._verbose)
                else:
                    out = None
                if self._verbose > 4:
                    t = time.time() - t0
                    _, name = get_func_name(self.func)
                    msg = '%s cache loaded - %s' % (name, format_time(t))
                    print(max(0, 80 - len(msg)) * '_' + msg)
            except Exception:
                _, signature = format_signature(self.func, *args, **kwargs)
                self.warn('Exception while loading results for {}\n {}'.format(signature, traceback.format_exc()))
                must_call = True
        else:
            if self._verbose > 10:
                _, name = get_func_name(self.func)
                self.warn('Computing func {0}, argument hash {1} in location {2}'.format(name, args_id, self.store_backend.get_cached_func_info([func_id])['location']))
            must_call = True
        if must_call:
            out, metadata = self.call(*args, **kwargs)
            if self.mmap_mode is not None:
                if self._verbose:
                    msg = _format_load_msg(func_id, args_id, timestamp=self.timestamp, metadata=metadata)
                out = self.store_backend.load_item([func_id, args_id], msg=msg, verbose=self._verbose)
        return (out, args_id, metadata)

    @property
    def func_code_info(self):
        if hasattr(self.func, '__code__'):
            if self._func_code_id is None:
                self._func_code_id = id(self.func.__code__)
            elif id(self.func.__code__) != self._func_code_id:
                self._func_code_info = None
        if self._func_code_info is None:
            self._func_code_info = get_func_code(self.func)
        return self._func_code_info

    def call_and_shelve(self, *args, **kwargs):
        """Call wrapped function, cache result and return a reference.

        This method returns a reference to the cached result instead of the
        result itself. The reference object is small and pickeable, allowing
        to send or store it easily. Call .get() on reference object to get
        result.

        Returns
        -------
        cached_result: MemorizedResult or NotMemorizedResult
            reference to the value returned by the wrapped function. The
            class "NotMemorizedResult" is used when there is no cache
            activated (e.g. location=None in Memory).
        """
        _, args_id, metadata = self._cached_call(args, kwargs, shelving=True)
        return MemorizedResult(self.store_backend, self.func, args_id, metadata=metadata, verbose=self._verbose - 1, timestamp=self.timestamp)

    def __call__(self, *args, **kwargs):
        return self._cached_call(args, kwargs)[0]

    def __getstate__(self):
        _ = self.func_code_info
        state = self.__dict__.copy()
        state['timestamp'] = None
        state['_func_code_id'] = None
        return state

    def check_call_in_cache(self, *args, **kwargs):
        """Check if function call is in the memory cache.

        Does not call the function or do any work besides func inspection
        and arg hashing.

        Returns
        -------
        is_call_in_cache: bool
            Whether or not the result of the function has been cached
            for the input arguments that have been passed.
        """
        func_id, args_id = self._get_output_identifiers(*args, **kwargs)
        return self.store_backend.contains_item((func_id, args_id))

    def _get_argument_hash(self, *args, **kwargs):
        return hashing.hash(filter_args(self.func, self.ignore, args, kwargs), coerce_mmap=self.mmap_mode is not None)

    def _get_output_identifiers(self, *args, **kwargs):
        """Return the func identifier and input parameter hash of a result."""
        func_id = _build_func_identifier(self.func)
        argument_hash = self._get_argument_hash(*args, **kwargs)
        return (func_id, argument_hash)

    def _hash_func(self):
        """Hash a function to key the online cache"""
        func_code_h = hash(getattr(self.func, '__code__', None))
        return (id(self.func), hash(self.func), func_code_h)

    def _write_func_code(self, func_code, first_line):
        """ Write the function code and the filename to a file.
        """
        func_id = _build_func_identifier(self.func)
        func_code = u'%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
        self.store_backend.store_cached_func_code([func_id], func_code)
        is_named_callable = False
        is_named_callable = hasattr(self.func, '__name__') and self.func.__name__ != '<lambda>'
        if is_named_callable:
            func_hash = self._hash_func()
            try:
                _FUNCTION_HASHES[self.func] = func_hash
            except TypeError:
                pass

    def _check_previous_func_code(self, stacklevel=2):
        """
            stacklevel is the depth a which this function is called, to
            issue useful warnings to the user.
        """
        try:
            if self.func in _FUNCTION_HASHES:
                func_hash = self._hash_func()
                if func_hash == _FUNCTION_HASHES[self.func]:
                    return True
        except TypeError:
            pass
        func_code, source_file, first_line = self.func_code_info
        func_id = _build_func_identifier(self.func)
        try:
            old_func_code, old_first_line = extract_first_line(self.store_backend.get_cached_func_code([func_id]))
        except (IOError, OSError):
            self._write_func_code(func_code, first_line)
            return False
        if old_func_code == func_code:
            return True
        _, func_name = get_func_name(self.func, resolv_alias=False, win_characters=False)
        if old_first_line == first_line == -1 or func_name == '<lambda>':
            if not first_line == -1:
                func_description = '{0} ({1}:{2})'.format(func_name, source_file, first_line)
            else:
                func_description = func_name
            warnings.warn(JobLibCollisionWarning("Cannot detect name collisions for function '{0}'".format(func_description)), stacklevel=stacklevel)
        if not old_first_line == first_line and source_file is not None:
            possible_collision = False
            if os.path.exists(source_file):
                _, func_name = get_func_name(self.func, resolv_alias=False)
                num_lines = len(func_code.split('\n'))
                with open_py_source(source_file) as f:
                    on_disk_func_code = f.readlines()[old_first_line - 1:old_first_line - 1 + num_lines - 1]
                on_disk_func_code = ''.join(on_disk_func_code)
                possible_collision = on_disk_func_code.rstrip() == old_func_code.rstrip()
            else:
                possible_collision = source_file.startswith('<doctest ')
            if possible_collision:
                warnings.warn(JobLibCollisionWarning("Possible name collisions between functions '%s' (%s:%i) and '%s' (%s:%i)" % (func_name, source_file, old_first_line, func_name, source_file, first_line)), stacklevel=stacklevel)
        if self._verbose > 10:
            _, func_name = get_func_name(self.func, resolv_alias=False)
            self.warn('Function {0} (identified by {1}) has changed.'.format(func_name, func_id))
        self.clear(warn=True)
        return False

    def clear(self, warn=True):
        """Empty the function's cache."""
        func_id = _build_func_identifier(self.func)
        if self._verbose > 0 and warn:
            self.warn('Clearing function cache identified by %s' % func_id)
        self.store_backend.clear_path([func_id])
        func_code, _, first_line = self.func_code_info
        self._write_func_code(func_code, first_line)

    def call(self, *args, **kwargs):
        """Force the execution of the function with the given arguments.

        The output values will be persisted, i.e., the cache will be updated
        with any new values.

        Parameters
        ----------
        *args: arguments
            The arguments.
        **kwargs: keyword arguments
            Keyword arguments.

        Returns
        -------
        output : object
            The output of the function call.
        metadata : dict
            The metadata associated with the call.
        """
        start_time = time.time()
        func_id, args_id = self._get_output_identifiers(*args, **kwargs)
        if self._verbose > 0:
            print(format_call(self.func, args, kwargs))
        output = self.func(*args, **kwargs)
        self.store_backend.dump_item([func_id, args_id], output, verbose=self._verbose)
        duration = time.time() - start_time
        metadata = self._persist_input(duration, args, kwargs)
        if self._verbose > 0:
            _, name = get_func_name(self.func)
            msg = '%s - %s' % (name, format_time(duration))
            print(max(0, 80 - len(msg)) * '_' + msg)
        return (output, metadata)

    def _persist_input(self, duration, args, kwargs, this_duration_limit=0.5):
        """ Save a small summary of the call using json format in the
            output directory.

            output_dir: string
                directory where to write metadata.

            duration: float
                time taken by hashing input arguments, calling the wrapped
                function and persisting its output.

            args, kwargs: list and dict
                input arguments for wrapped function

            this_duration_limit: float
                Max execution time for this function before issuing a warning.
        """
        start_time = time.time()
        argument_dict = filter_args(self.func, self.ignore, args, kwargs)
        input_repr = dict(((k, repr(v)) for k, v in argument_dict.items()))
        metadata = {'duration': duration, 'input_args': input_repr, 'time': start_time}
        func_id, args_id = self._get_output_identifiers(*args, **kwargs)
        self.store_backend.store_metadata([func_id, args_id], metadata)
        this_duration = time.time() - start_time
        if this_duration > this_duration_limit:
            warnings.warn('Persisting input arguments took %.2fs to run.If this happens often in your code, it can cause performance problems (results will be correct in all cases). The reason for this is probably some large input arguments for a wrapped function.' % this_duration, stacklevel=5)
        return metadata

    def __repr__(self):
        return '{class_name}(func={func}, location={location})'.format(class_name=self.__class__.__name__, func=self.func, location=self.store_backend.location)