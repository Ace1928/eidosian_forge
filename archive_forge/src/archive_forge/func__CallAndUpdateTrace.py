from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import inspect
import json
import os
import pipes
import re
import shlex
import sys
import types
from fire import completion
from fire import decorators
from fire import formatting
from fire import helptext
from fire import inspectutils
from fire import interact
from fire import parser
from fire import trace
from fire import value_types
from fire.console import console_io
import six
def _CallAndUpdateTrace(component, args, component_trace, treatment='class', target=None):
    """Call the component by consuming args from args, and update the FireTrace.

  The component could be a class, a routine, or a callable object. This function
  calls the component and adds the appropriate action to component_trace.

  Args:
    component: The component to call
    args: Args for calling the component
    component_trace: FireTrace object that contains action trace
    treatment: Type of treatment used. Indicating whether we treat the component
        as a class, a routine, or a callable.
    target: Target in FireTrace element, default is None. If the value is None,
        the component itself will be used as target.
  Returns:
    component: The object that is the result of the callable call.
    remaining_args: The remaining args that haven't been consumed yet.
  """
    if not target:
        target = component
    filename, lineno = inspectutils.GetFileAndLine(component)
    metadata = decorators.GetMetadata(component)
    fn = component.__call__ if treatment == 'callable' else component
    parse = _MakeParseFn(fn, metadata)
    (varargs, kwargs), consumed_args, remaining_args, capacity = parse(args)
    if inspectutils.IsCoroutineFunction(fn):
        loop = asyncio.get_event_loop()
        component = loop.run_until_complete(fn(*varargs, **kwargs))
    else:
        component = fn(*varargs, **kwargs)
    if treatment == 'class':
        action = trace.INSTANTIATED_CLASS
    elif treatment == 'routine':
        action = trace.CALLED_ROUTINE
    else:
        action = trace.CALLED_CALLABLE
    component_trace.AddCalledComponent(component, target, consumed_args, filename, lineno, capacity, action=action)
    return (component, remaining_args)