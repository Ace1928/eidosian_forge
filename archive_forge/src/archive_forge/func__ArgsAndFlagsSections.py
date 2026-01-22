from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import itertools
import sys
from fire import completion
from fire import custom_descriptions
from fire import decorators
from fire import docstrings
from fire import formatting
from fire import inspectutils
from fire import value_types
def _ArgsAndFlagsSections(info, spec, metadata):
    """The "Args and Flags" sections of the help string."""
    args_with_no_defaults = spec.args[:len(spec.args) - len(spec.defaults)]
    args_with_defaults = spec.args[len(spec.args) - len(spec.defaults):]
    accepts_positional_args = metadata.get(decorators.ACCEPTS_POSITIONAL_ARGS)
    args_and_flags_sections = []
    notes_sections = []
    docstring_info = info['docstring_info']
    arg_items = [_CreateArgItem(arg, docstring_info, spec) for arg in args_with_no_defaults]
    if spec.varargs:
        arg_items.append(_CreateArgItem(spec.varargs, docstring_info, spec))
    if arg_items:
        title = 'POSITIONAL ARGUMENTS' if accepts_positional_args else 'ARGUMENTS'
        arguments_section = (title, '\n'.join(arg_items).rstrip('\n'))
        args_and_flags_sections.append(arguments_section)
        if args_with_no_defaults and accepts_positional_args:
            notes_sections.append(('NOTES', 'You can also use flags syntax for POSITIONAL ARGUMENTS'))
    unique_short_args = _GetShortFlags(args_with_defaults)
    positional_flag_items = [_CreateFlagItem(flag, docstring_info, spec, required=False, short_arg=flag[0] in unique_short_args) for flag in args_with_defaults]
    unique_short_kwonly_flags = _GetShortFlags(spec.kwonlyargs)
    kwonly_flag_items = [_CreateKeywordOnlyFlagItem(flag, docstring_info, spec, short_arg=flag[0] in unique_short_kwonly_flags) for flag in spec.kwonlyargs]
    flag_items = positional_flag_items + kwonly_flag_items
    if spec.varkw:
        documented_kwargs = []
        flag_string = '--{name}'
        short_flag_string = '-{short_name}, --{name}'
        flags = docstring_info.args or []
        flag_names = [f.name for f in flags]
        unique_short_flags = _GetShortFlags(flag_names)
        for flag in flags:
            if isinstance(flag, docstrings.KwargInfo):
                if flag.name[0] in unique_short_flags:
                    flag_string = short_flag_string.format(name=flag.name, short_name=flag.name[0])
                else:
                    flag_string = flag_string.format(name=flag.name)
                flag_item = _CreateFlagItem(flag.name, docstring_info, spec, flag_string=flag_string)
                documented_kwargs.append(flag_item)
        if documented_kwargs:
            if flag_items:
                message = 'The following flags are also accepted.'
                item = _CreateItem(message, None, indent=4)
                flag_items.append(item)
            flag_items.extend(documented_kwargs)
        description = _GetArgDescription(spec.varkw, docstring_info)
        if documented_kwargs:
            message = 'Additional undocumented flags may also be accepted.'
        elif flag_items:
            message = 'Additional flags are accepted.'
        else:
            message = 'Flags are accepted.'
        item = _CreateItem(message, description, indent=4)
        flag_items.append(item)
    if flag_items:
        flags_section = ('FLAGS', '\n'.join(flag_items))
        args_and_flags_sections.append(flags_section)
    return (args_and_flags_sections, notes_sections)