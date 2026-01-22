from __future__ import unicode_literals
import collections
import contextlib
import inspect
import logging
import pprint
import sys
import textwrap
import six
class ConfigObject(six.with_metaclass(ConfigMeta)):
    """
  Base class for encapsulationg options/parameters
  """
    _field_registry = []

    def _update_derived(self):
        """subclass hook to update any derived values after a change."""

    def __init__(self, **kwargs):
        assert hasattr(self.__class__, '_field_registry'), 'Derived class {} is missing _field_registry member'.format(self.__class__.__name__)
        self.consume_known(kwargs)
        self.legacy_consume(kwargs)
        warn_unused(kwargs)

    def consume_known(self, kwargs):
        """Consume known configuration values from a dictionary. Removes the
       values from the dictionary.
    """
        for descr in self._field_registry:
            if descr.name in kwargs:
                descr.consume_value(self, kwargs.pop(descr.name))
        self._update_derived()

    def legacy_consume(self, kwargs):
        """Consume arguments from the root of the configuration dictionary.
    """
        self.consume_known(kwargs)
        for descr in self._field_registry:
            if isinstance(descr, SubtreeDescriptor):
                descr.legacy_shim_consume(self, kwargs)

    @classmethod
    def get_field_names(cls):
        return [descr.name for descr in cls._field_registry]

    def _as_dict(self, dtype, with_help=False, with_defaults=True):
        """
    Return a dictionary mapping field names to their values only for fields
    specified in the constructor
    """
        out = dtype()
        for descr in self._field_registry:
            value = descr.__get__(self, type(self))
            if isinstance(descr, SubtreeDescriptor):
                if not (with_defaults or value.has_override()):
                    continue
                if with_help:
                    out['_help_' + descr.name] = serialize_docstring(value.__doc__)
                out[descr.name] = value._as_dict(dtype, with_help, with_defaults)
            elif isinstance(descr, FieldDescriptor):
                if not (with_defaults or descr.has_override(self)):
                    continue
                helptext = descr.helptext
                if helptext and with_help:
                    out['_help_' + descr.name] = textwrap.wrap(helptext, 60)
                out[descr.name] = serialize(value)
            else:
                raise RuntimeError('Field registry contains unknown descriptor')
        return out

    def as_dict(self, with_help=False, with_defaults=True):
        return self._as_dict(dict, with_help, with_defaults)

    def as_odict(self, with_help=False, with_defaults=True):
        return self._as_dict(collections.OrderedDict, with_help, with_defaults)

    def has_override(self):
        for descr in self._field_registry:
            if isinstance(descr, SubtreeDescriptor):
                if descr.get(self).has_override():
                    return True
            elif isinstance(descr, FieldDescriptor):
                if descr.has_override(self):
                    return True
            else:
                raise RuntimeError('Field registry contains unknown descriptor')
        return False

    def dump(self, outfile, depth=0, with_help=True, with_defaults=True):
        indent = '  ' * depth
        linewidth = 80 - len(indent)
        comment_linewidth = 80 - len(indent) - len('# ')
        ppr = pprint.PrettyPrinter(indent=2, width=linewidth)
        for descr in self._field_registry:
            fieldname = descr.name
            value = descr.__get__(self, type(self))
            if isinstance(descr, SubtreeDescriptor):
                if not (with_defaults or value.has_override()):
                    continue
                if value.__doc__ and with_help:
                    section_doc_lines = value.__doc__.split('\n')
                    ruler_len = max((len(line) for line in section_doc_lines))
                    outfile.write('# {}\n'.format('-' * ruler_len))
                    for line in section_doc_lines:
                        outfile.write(indent)
                        outfile.write('# {}\n'.format(line.rstrip()))
                    outfile.write('# {}\n'.format('-' * ruler_len))
                outfile.write(indent)
                outfile.write('with section("{}"):\n'.format(fieldname))
                value.dump(outfile, depth + 1, with_help, with_defaults)
                outfile.write('\n')
            elif isinstance(descr, FieldDescriptor):
                if not (with_defaults or descr.has_override(self)):
                    continue
                helptext = descr.helptext
                if helptext and with_help:
                    outfile.write('\n')
                    for line in textwrap.wrap(helptext, comment_linewidth):
                        outfile.write(indent)
                        outfile.write('# {}\n'.format(line.rstrip()))
                pretty = '{} = {}'.format(fieldname, ppr.pformat(value))
                for line in pretty.split('\n'):
                    outfile.write(indent)
                    outfile.write(line)
                    outfile.write('\n')
            else:
                raise RuntimeError('Field registry contains unknown descriptor')

    @classmethod
    def add_to_argparser(cls, argparser):
        if '\n' in cls.__doc__:
            title, description = cls.__doc__.split('\n', 1)
        else:
            title = cls.__doc__
            description = None
        optgroup = argparser.add_argument_group(title=title, description=description)
        for descr in cls._field_registry:
            if isinstance(descr, SubtreeDescriptor):
                descr.add_to_argparser(argparser)
            elif isinstance(descr, FieldDescriptor):
                descr.add_to_argparse(optgroup)

    def clone(self):
        """
    Return a copy of self.
    """
        return self.__class__(**self.as_dict())

    def validate(self):
        return True