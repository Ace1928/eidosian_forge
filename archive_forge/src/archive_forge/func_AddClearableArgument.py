from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def AddClearableArgument(parser, name, set_help, clear_help, dest=None, **kwargs):
    """Adds arguments to `parser` for modifying or clearing a text field.

  A generic implementation of the style guidelines at
  go/gcloud-style#createupdate-command-flags.

  Args:
    parser: the ArgumentParser to which the arguments will be added.
    name: name of the field to be modified.
    set_help: help text for the argument that sets the field. Should explain
      what the field itself is.
    clear_help: help text for the argument that clears the field.
    dest: suffix for destiantion fields.
    **kwargs: additional parameters for the setter argument.
  """
    mutex_group = parser.add_mutually_exclusive_group()
    mutex_group.add_argument('--' + name, help=set_help, dest='set_' + (dest or name.replace('-', '_')), **kwargs)
    mutex_group.add_argument('--clear-' + name, dest='clear_' + (dest or name.replace('-', '_')), help=clear_help, action='store_true')