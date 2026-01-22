from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import functools
from absl._collections_abc import abc
from absl.flags import _argument_parser
from absl.flags import _exceptions
from absl.flags import _helpers
import six
class MultiFlag(Flag):
    """A flag that can appear multiple time on the command-line.

  The value of such a flag is a list that contains the individual values
  from all the appearances of that flag on the command-line.

  See the __doc__ for Flag for most behavior of this class.  Only
  differences in behavior are described here:

    * The default value may be either a single value or an iterable of values.
      A single value is transformed into a single-item list of that value.

    * The value of the flag is always a list, even if the option was
      only supplied once, and even if the default value is a single
      value
  """

    def __init__(self, *args, **kwargs):
        super(MultiFlag, self).__init__(*args, **kwargs)
        self.help += ';\n    repeat this option to specify a list of values'

    def parse(self, arguments):
        """Parses one or more arguments with the installed parser.

    Args:
      arguments: a single argument or a list of arguments (typically a
        list of default values); a single argument is converted
        internally into a list containing one item.
    """
        new_values = self._parse(arguments)
        if self.present:
            self.value.extend(new_values)
        else:
            self.value = new_values
        self.present += len(new_values)

    def _parse(self, arguments):
        if isinstance(arguments, abc.Iterable) and (not isinstance(arguments, six.string_types)):
            arguments = list(arguments)
        if not isinstance(arguments, list):
            arguments = [arguments]
        return [super(MultiFlag, self)._parse(item) for item in arguments]

    def _serialize(self, value):
        """See base class."""
        if not self.serializer:
            raise _exceptions.Error('Serializer not present for flag %s' % self.name)
        if value is None:
            return ''
        serialized_items = [super(MultiFlag, self)._serialize(value_item) for value_item in value]
        return '\n'.join(serialized_items)

    def flag_type(self):
        """See base class."""
        return 'multi ' + self.parser.flag_type()

    def _extra_xml_dom_elements(self, doc):
        elements = []
        if hasattr(self.parser, 'enum_values'):
            for enum_value in self.parser.enum_values:
                elements.append(_helpers.create_xml_dom_element(doc, 'enum_value', enum_value))
        return elements