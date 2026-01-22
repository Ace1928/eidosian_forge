from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import shlex
class ExampleCommandMetadata(object):
    """Encapsulates metadata about entire example command string.

  Attributes:
    argument_metadatas: A list containing the metadata for each argument in an
    example command string.
  """

    def __init__(self):
        self._argument_metadatas = []

    def add_arg_metadata(self, arg_metadata):
        """Adds an argument's metadata to comprehensive metadata list.

    Args:
      arg_metadata: The argument metadata to be added.
    """
        self._argument_metadatas.append(arg_metadata)

    def get_argument_metadatas(self):
        """Returns the metadata for an entire example command string."""
        return sorted(self._argument_metadatas, key=lambda x: x.stop_index)

    def __eq__(self, other):
        if isinstance(other, ExampleCommandMetadata):
            our_data = sorted(self._argument_metadatas, key=lambda x: x.stop_index)
            other_data = sorted(other._argument_metadatas, key=lambda x: x.stop_index)
            if len(our_data) != len(other_data):
                return False
            for i in range(len(self._argument_metadatas)):
                if our_data[i] != other_data[i]:
                    return False
            return True
        return False

    def __ne__(self, other):
        return self.__eq__(other)

    def __str__(self):
        metadatas = self.get_argument_metadatas()
        result = [str(data) for data in metadatas]
        return '[{result}]'.format(result=',  '.join(result))