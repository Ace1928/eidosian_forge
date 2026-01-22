import traceback
from collections import namedtuple, defaultdict
import itertools
import logging
import textwrap
from shutil import get_terminal_size
from .abstract import Callable, DTypeSpec, Dummy, Literal, Type, weakref
from .common import Opaque
from .misc import unliteral
from numba.core import errors, utils, types, config
from numba.core.typeconv import Conversion
class _ResolutionFailures(object):
    """Collect and format function resolution failures.
    """

    def __init__(self, context, function_type, args, kwargs, depth=0):
        self._context = context
        self._function_type = function_type
        self._args = args
        self._kwargs = kwargs
        self._failures = defaultdict(list)
        self._depth = depth
        self._max_depth = 5
        self._scale = 2

    def __len__(self):
        return len(self._failures)

    def add_error(self, calltemplate, matched, error, literal):
        """
        Args
        ----
        calltemplate : CallTemplate
        error : Exception or str
            Error message
        """
        isexc = isinstance(error, Exception)
        errclazz = '%s: ' % type(error).__name__ if isexc else ''
        key = '{}{}'.format(errclazz, str(error))
        self._failures[key].append(_FAILURE(calltemplate, matched, error, literal))

    def format(self):
        """Return a formatted error message from all the gathered errors.
        """
        indent = ' ' * self._scale
        argstr = argsnkwargs_to_str(self._args, self._kwargs)
        ncandidates = sum([len(x) for x in self._failures.values()])
        tykey = self._function_type.typing_key
        fname = getattr(tykey, '__name__', None)
        is_external_fn_ptr = isinstance(self._function_type, ExternalFunctionPointer)
        if fname is None:
            if is_external_fn_ptr:
                fname = 'ExternalFunctionPointer'
            else:
                fname = '<unknown function>'
        msgbuf = [_header_template.format(the_function=self._function_type, fname=fname, signature=argstr, ncandidates=ncandidates)]
        nolitargs = tuple([unliteral(a) for a in self._args])
        nolitkwargs = {k: unliteral(v) for k, v in self._kwargs.items()}
        nolitargstr = argsnkwargs_to_str(nolitargs, nolitkwargs)
        ldepth = min(max(self._depth, 0), self._max_depth)

        def template_info(tp):
            src_info = tp.get_template_info()
            unknown = 'unknown'
            source_name = src_info.get('name', unknown)
            source_file = src_info.get('filename', unknown)
            source_lines = src_info.get('lines', unknown)
            source_kind = src_info.get('kind', 'Unknown template')
            return (source_name, source_file, source_lines, source_kind)
        for i, (k, err_list) in enumerate(self._failures.items()):
            err = err_list[0]
            nduplicates = len(err_list)
            template, error = (err.template, err.error)
            ifo = template_info(template)
            source_name, source_file, source_lines, source_kind = ifo
            largstr = argstr if err.literal else nolitargstr
            if err.error == 'No match.':
                err_dict = defaultdict(set)
                for errs in err_list:
                    err_dict[errs.template].add(errs.literal)
                if len(err_dict) == 1:
                    template = [_ for _ in err_dict.keys()][0]
                    source_name, source_file, source_lines, source_kind = template_info(template)
                    source_lines = source_lines[0]
                else:
                    source_file = '<numerous>'
                    source_lines = 'N/A'
                msgbuf.append(_termcolor.errmsg(_wrapper(_overload_template.format(nduplicates=nduplicates, kind=source_kind.title(), function=fname, inof='of', file=source_file, line=source_lines, args=largstr), ldepth + 1)))
                msgbuf.append(_termcolor.highlight(_wrapper(err.error, ldepth + 2)))
            else:
                msgbuf.append(_termcolor.errmsg(_wrapper(_overload_template.format(nduplicates=nduplicates, kind=source_kind.title(), function=source_name, inof='in', file=source_file, line=source_lines[0], args=largstr), ldepth + 1)))
                if isinstance(error, BaseException):
                    reason = indent + self.format_error(error)
                    errstr = _err_reasons['specific_error'].format(reason)
                else:
                    errstr = error
                if config.DEVELOPER_MODE:
                    if isinstance(error, BaseException):
                        bt = traceback.format_exception(type(error), error, error.__traceback__)
                    else:
                        bt = ['']
                    bt_as_lines = _bt_as_lines(bt)
                    nd2indent = '\n{}'.format(2 * indent)
                    errstr += _termcolor.reset(nd2indent + nd2indent.join(bt_as_lines))
                msgbuf.append(_termcolor.highlight(_wrapper(errstr, ldepth + 2)))
                loc = self.get_loc(template, error)
                if loc:
                    msgbuf.append('{}raised from {}'.format(indent, loc))
        return _wrapper('\n'.join(msgbuf) + '\n')

    def format_error(self, error):
        """Format error message or exception
        """
        if isinstance(error, Exception):
            return '{}: {}'.format(type(error).__name__, error)
        else:
            return '{}'.format(error)

    def get_loc(self, classtemplate, error):
        """Get source location information from the error message.
        """
        if isinstance(error, Exception) and hasattr(error, '__traceback__'):
            frame = traceback.extract_tb(error.__traceback__)[-1]
            return '{}:{}'.format(frame[0], frame[1])

    def raise_error(self):
        for faillist in self._failures.values():
            for fail in faillist:
                if isinstance(fail.error, errors.ForceLiteralArg):
                    raise fail.error
        raise errors.TypingError(self.format())