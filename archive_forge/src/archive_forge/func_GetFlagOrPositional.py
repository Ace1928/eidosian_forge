from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.core.util import files
def GetFlagOrPositional(name, positional=False, **kwargs):
    """Return argument called name as either flag or positional."""
    dest = name.replace('-', '_').upper()
    if positional:
        flag = dest
        kwargs.pop('required', None)
    else:
        flag = '--{}'.format(name.replace('_', '-').lower())
    if not positional:
        kwargs['dest'] = dest
    return base.Argument(flag, **kwargs)