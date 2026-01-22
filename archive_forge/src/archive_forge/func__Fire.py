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
def _Fire(component, args, parsed_flag_args, context, name=None):
    """Execute a Fire command on a target component using the args supplied.

  Arguments that come after a final isolated '--' are treated as Flags, eg for
  interactive mode or completion script generation.

  Other arguments are consumed by the execution of the Fire command, eg in the
  traversal of the members of the component, or in calling a function or
  instantiating a class found during the traversal.

  The steps performed by this method are:

  1. Parse any Flag args (the args after the final --)

  2. Start with component as the current component.
  2a. If the current component is a class, instantiate it using args from args.
  2b. If the component is a routine, call it using args from args.
  2c. If the component is a sequence, index into it using an arg from
      args.
  2d. If possible, access a member from the component using an arg from args.
  2e. If the component is a callable object, call it using args from args.
  2f. Repeat 2a-2e until no args remain.
  Note: Only the first applicable rule from 2a-2e is applied in each iteration.
  After each iteration of step 2a-2e, the current component is updated to be the
  result of the applied rule.

  3a. Embed into ipython REPL if interactive mode is selected.
  3b. Generate a completion script if that flag is provided.

  In step 2, arguments will only ever be consumed up to a separator; a single
  step will never consume arguments from both sides of a separator.
  The separator defaults to a hyphen (-), and can be overwritten with the
  --separator Fire argument.

  Args:
    component: The target component for Fire.
    args: A list of args to consume in Firing on the component, usually from
        the command line.
    parsed_flag_args: The values of the flag args (e.g. --verbose, --separator)
        that are part of every Fire CLI.
    context: A dict with the local and global variables available at the call
        to Fire.
    name: Optional. The name of the command. Used in interactive mode and in
        the tab completion script.
  Returns:
    FireTrace of components starting with component, tracing Fire's execution
        path as it consumes args.
  Raises:
    ValueError: If there are arguments that cannot be consumed.
    ValueError: If --completion is specified but no name available.
  """
    verbose = parsed_flag_args.verbose
    interactive = parsed_flag_args.interactive
    separator = parsed_flag_args.separator
    show_completion = parsed_flag_args.completion
    show_help = parsed_flag_args.help
    show_trace = parsed_flag_args.trace
    if component is None:
        component = context
    initial_component = component
    component_trace = trace.FireTrace(initial_component=initial_component, name=name, separator=separator, verbose=verbose, show_help=show_help, show_trace=show_trace)
    instance = None
    remaining_args = args
    while True:
        last_component = component
        initial_args = remaining_args
        if not remaining_args and (show_help or interactive or show_trace or (show_completion is not None)):
            break
        if _IsHelpShortcut(component_trace, remaining_args):
            remaining_args = []
            break
        saved_args = []
        used_separator = False
        if separator in remaining_args:
            separator_index = remaining_args.index(separator)
            saved_args = remaining_args[separator_index + 1:]
            remaining_args = remaining_args[:separator_index]
            used_separator = True
        assert separator not in remaining_args
        handled = False
        candidate_errors = []
        is_callable = inspect.isclass(component) or inspect.isroutine(component)
        is_callable_object = callable(component) and (not is_callable)
        is_sequence = isinstance(component, (list, tuple))
        is_map = isinstance(component, dict) or inspectutils.IsNamedTuple(component)
        if not handled and is_callable:
            is_class = inspect.isclass(component)
            try:
                component, remaining_args = _CallAndUpdateTrace(component, remaining_args, component_trace, treatment='class' if is_class else 'routine', target=component.__name__)
                handled = True
            except FireError as error:
                candidate_errors.append((error, initial_args))
            if handled and last_component is initial_component:
                instance = component
        if not handled and is_sequence and remaining_args:
            arg = remaining_args[0]
            try:
                index = int(arg)
                component = component[index]
                handled = True
            except (ValueError, IndexError):
                error = FireError('Unable to index into component with argument:', arg)
                candidate_errors.append((error, initial_args))
            if handled:
                remaining_args = remaining_args[1:]
                filename = None
                lineno = None
                component_trace.AddAccessedProperty(component, index, [arg], filename, lineno)
        if not handled and is_map and remaining_args:
            target = remaining_args[0]
            if inspectutils.IsNamedTuple(component):
                component_dict = component._asdict()
            else:
                component_dict = component
            if target in component_dict:
                component = component_dict[target]
                handled = True
            elif target.replace('-', '_') in component_dict:
                component = component_dict[target.replace('-', '_')]
                handled = True
            else:
                for key, value in component_dict.items():
                    if target == str(key):
                        component = value
                        handled = True
                        break
            if handled:
                remaining_args = remaining_args[1:]
                filename = None
                lineno = None
                component_trace.AddAccessedProperty(component, target, [target], filename, lineno)
            else:
                error = FireError('Cannot find key:', target)
                candidate_errors.append((error, initial_args))
        if not handled and remaining_args:
            try:
                target = remaining_args[0]
                component, consumed_args, remaining_args = _GetMember(component, remaining_args)
                handled = True
                filename, lineno = inspectutils.GetFileAndLine(component)
                component_trace.AddAccessedProperty(component, target, consumed_args, filename, lineno)
            except FireError as error:
                candidate_errors.append((error, initial_args))
        if not handled and is_callable_object:
            try:
                component, remaining_args = _CallAndUpdateTrace(component, remaining_args, component_trace, treatment='callable')
                handled = True
            except FireError as error:
                candidate_errors.append((error, initial_args))
        if not handled and candidate_errors:
            error, initial_args = candidate_errors[0]
            component_trace.AddError(error, initial_args)
            return component_trace
        if used_separator:
            if remaining_args:
                remaining_args = remaining_args + [separator] + saved_args
            elif inspect.isclass(last_component) or inspect.isroutine(last_component):
                remaining_args = saved_args
                component_trace.AddSeparator()
            elif component is not last_component:
                remaining_args = [separator] + saved_args
            else:
                remaining_args = saved_args
        if component is last_component and remaining_args == initial_args:
            break
    if remaining_args:
        component_trace.AddError(FireError('Could not consume arguments:', remaining_args), initial_args)
        return component_trace
    if show_completion is not None:
        if name is None:
            raise ValueError('Cannot make completion script without command name')
        script = CompletionScript(name, initial_component, shell=show_completion)
        component_trace.AddCompletionScript(script)
    if interactive:
        variables = context.copy()
        if name is not None:
            variables[name] = initial_component
        variables['component'] = initial_component
        variables['result'] = component
        variables['trace'] = component_trace
        if instance is not None:
            variables['self'] = instance
        interact.Embed(variables, verbose)
        component_trace.AddInteractiveMode()
    return component_trace