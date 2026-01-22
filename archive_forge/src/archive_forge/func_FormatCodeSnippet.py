from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import six
def FormatCodeSnippet(arg_name, arg_value, append=False):
    """Formats flag in markdown code snippet.

  Args:
    arg_name: str, name of the flag in snippet
    arg_value: str, flag value in snippet
    append: bool, whether to use append syntax for flag

  Returns:
    markdown string of example user input
  """
    if ' ' in arg_value:
        example_flag = "{}='{}'".format(arg_name, arg_value)
    else:
        example_flag = '{}={}'.format(arg_name, arg_value)
    if append:
        return '```\n\n{input} {input}\n\n```'.format(input=example_flag)
    else:
        return '```\n\n{}\n\n```'.format(example_flag)