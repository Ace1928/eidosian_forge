import collections
import datetime
import functools
import inspect
import socket
import threading
from oslo_utils import reflection
from oslo_utils import uuidutils
from osprofiler import _utils as utils
from osprofiler import notifier
def trace_cls(name, info=None, hide_args=False, hide_result=True, trace_private=False, allow_multiple_trace=True, trace_class_methods=False, trace_static_methods=False):
    """Trace decorator for instances of class .

    Very useful if you would like to add trace point on existing method:

    >>  @profiler.trace_cls("rpc")
    >>  RpcManagerClass(object):
    >>
    >>      def my_method(self, some_args):
    >>          pass
    >>
    >>      def my_method2(self, some_arg1, some_arg2, kw=None, kw2=None)
    >>          pass
    >>

    :param name: The name of action. E.g. wsgi, rpc, db, etc..
    :param info: Dictionary with extra trace information. For example in wsgi
                 it can be url, in rpc - message or in db sql - request.
    :param hide_args: Don't push to trace info args and kwargs. Quite useful
                      if you have some info in args that you wont to share,
                      e.g. passwords.
    :param hide_result: Boolean value to hide/show function result in trace.
                        True - hide function result (default).
                        False - show function result in trace.
    :param trace_private: Trace methods that starts with "_". It wont trace
                          methods that starts "__" even if it is turned on.
    :param trace_static_methods: Trace staticmethods. This may be prone to
                                 issues so careful usage is recommended (this
                                 is also why this defaults to false).
    :param trace_class_methods: Trace classmethods. This may be prone to
                                issues so careful usage is recommended (this
                                is also why this defaults to false).
    :param allow_multiple_trace: If wrapped attributes have already been
                                 traced either allow the new trace to occur
                                 or raise a value error denoting that multiple
                                 tracing is not allowed (by default allow).
    """

    def trace_checker(attr_name, to_be_wrapped):
        if attr_name.startswith('__'):
            return (False, None)
        if not trace_private and attr_name.startswith('_'):
            return (False, None)
        if isinstance(to_be_wrapped, staticmethod):
            if not trace_static_methods:
                return (False, None)
            return (True, staticmethod)
        if isinstance(to_be_wrapped, classmethod):
            if not trace_class_methods:
                return (False, None)
            return (True, classmethod)
        return (True, None)

    def decorator(cls):
        clss = cls if inspect.isclass(cls) else cls.__class__
        mro_dicts = [c.__dict__ for c in inspect.getmro(clss)]
        traceable_attrs = []
        traceable_wrappers = []
        for attr_name, attr in inspect.getmembers(cls):
            if not (inspect.ismethod(attr) or inspect.isfunction(attr)):
                continue
            wrapped_obj = None
            for cls_dict in mro_dicts:
                if attr_name in cls_dict:
                    wrapped_obj = cls_dict[attr_name]
                    break
            should_wrap, wrapper = trace_checker(attr_name, wrapped_obj)
            if not should_wrap:
                continue
            traceable_attrs.append((attr_name, attr))
            traceable_wrappers.append(wrapper)
        if not allow_multiple_trace:
            _ensure_no_multiple_traced(traceable_attrs)
        for i, (attr_name, attr) in enumerate(traceable_attrs):
            wrapped_method = trace(name, info=info, hide_args=hide_args, hide_result=hide_result)(attr)
            wrapper = traceable_wrappers[i]
            if wrapper is not None:
                wrapped_method = wrapper(wrapped_method)
            setattr(cls, attr_name, wrapped_method)
        return cls
    return decorator