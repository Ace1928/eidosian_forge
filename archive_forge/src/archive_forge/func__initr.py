import enum
import logging
import os
import sys
import typing
import warnings
from rpy2.rinterface_lib import openrlib
from rpy2.rinterface_lib import callbacks
def _initr(interactive: typing.Optional[bool]=None, _want_setcallbacks: bool=True, _c_stack_limit: typing.Optional[int]=None) -> typing.Optional[int]:
    """Initialize the embedded R.

    :param interactive: Should R run in interactive or non-interactive mode?
    if `None` the value in `_DEFAULT_R_INTERACTIVE` will be used.
    :param _want_setcallbacks: Should custom rpy2 callbacks for R frontends
    be set?.
    :param _c_stack_limit: Limit for the C Stack.
    if `None` the value in `_DEFAULT_C_STACK_LIMIT` will be used.
    """
    if interactive is None:
        interactive = _DEFAULT_R_INTERACTIVE
    if _c_stack_limit is None:
        _c_stack_limit = _DEFAULT_C_STACK_LIMIT
    rlib = openrlib.rlib
    ffi_proxy = openrlib.ffi_proxy
    if ffi_proxy.get_ffi_mode(openrlib._rinterface_cffi) == ffi_proxy.InterfaceType.ABI:
        callback_funcs = callbacks
    else:
        callback_funcs = rlib
    with openrlib.rlock:
        if isinitialized():
            logger.info('R is already initialized. No need to initialize.')
            return None
        elif openrlib.R_HOME is None:
            raise ValueError('openrlib.R_HOME cannot be None.')
        elif openrlib.rlib.R_NilValue != ffi.NULL:
            msg = 'R was initialized outside of rpy2 (R_NilValue != NULL). Trying to use it nevertheless.'
            warnings.warn(msg)
            logger.warn(msg)
            _setinitialized()
            return None
        os.environ['R_HOME'] = openrlib.R_HOME
        os.environ['LD_LIBRARY_PATH'] = ':'.join((openrlib.LD_LIBRARY_PATH, os.environ.get('LD_LIBRARY_PATH', '')))
        options_c = [ffi.new('char[]', o.encode('ASCII')) for o in _options]
        n_options = len(options_c)
        n_options_c = ffi.cast('int', n_options)
        rlib.R_SignalHandlers = 0
        rlib.Rf_initialize_R(n_options_c, options_c)
        if _c_stack_limit:
            rlib.R_CStackLimit = ffi.cast('uintptr_t', _c_stack_limit)
        rlib.R_Interactive = True
        logger.debug('Calling R setup_Rmainloop.')
        rlib.setup_Rmainloop()
        _setinitialized()
        rlib.R_Interactive = interactive
        rlib.R_Outputfile = ffi.NULL
        rlib.R_Consolefile = ffi.NULL
        if _want_setcallbacks:
            logger.debug('Setting functions for R callbacks.')
            for rlib_symbol, callback_symbol in CALLBACK_INIT_PAIRS:
                _setcallback(rlib, rlib_symbol, callback_funcs, callback_symbol)
    return 1