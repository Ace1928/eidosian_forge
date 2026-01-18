import argparse
import ast
import re
import sys
def validate_slicing_string(slicing_string):
    """Validate a slicing string.

  Check if the input string contains only brackets, digits, commas and
  colons that are valid characters in numpy-style array slicing.

  Args:
    slicing_string: (str) Input slicing string to be validated.

  Returns:
    (bool) True if and only if the slicing string is valid.
  """
    return bool(re.search('^\\[(\\d|,|\\s|:)+\\]$', slicing_string))