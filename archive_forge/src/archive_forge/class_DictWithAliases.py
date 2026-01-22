from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
class DictWithAliases(dict):
    """A dict intended for serialized results which need computed values.

  DictWithAliases behaves like a dictionary with the exception of containing
  a MakeSerializable hook which excludes the "aliases" key if present in the
  dictionary from being returned. This is to allow additional pieces of data
  to be stored in the object which will not be output via the structured
  --format types for the commands.

  Example:
  d = DictWithAliases({'key': 'value', 'aliases': {'foo': 'bar'}})
  print(d['aliases']['foo']) # prints 'bar'
  d.MakeSeralizable() # returns {'key': 'value'}
  """

    def MakeSerializable(self):
        """Returns the underlying data without the aliases key for serialization."""
        data = self.copy()
        data.pop(_ROOT_ALIAS_KEY, None)
        return data

    def AddAlias(self, key, value):
        self.setdefault(_ROOT_ALIAS_KEY, {})[key] = value