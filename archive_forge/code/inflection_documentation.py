import re
import unicodedata
Make an underscored, lowercase form from the expression in the string.

  Example::

      >>> underscore("DeviceType")
      "device_type"

  As a rule of thumb you can think of :func:`underscore` as the inverse of
  :func:`camelize`, though there are cases where that does not hold::

      >>> camelize(underscore("IOError"))
      "IoError"

  Args:
    word: (str) A word to make underscored.

  Returns:
    A string with underscores.
  