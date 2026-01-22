from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections
import enum
import functools
import six
class ClassWithMultilineDocstring(object):
    """Test class for testing help text output with multiline docstring.

  This is a test class that has a long docstring description that spans across
  multiple lines for testing line breaking in help text.
  """

    @staticmethod
    def example_generator(n):
        """Generators have a ``Yields`` section instead of a ``Returns`` section.

    Args:
        n (int): The upper limit of the range to generate, from 0 to `n` - 1.

    Yields:
        int: The next number in the range of 0 to `n` - 1.

    Examples:
        Examples should be written in doctest format, and should illustrate how
        to use the function.

        >>> print([i for i in example_generator(4)])
        [0, 1, 2, 3]

    """
        for i in range(n):
            yield i