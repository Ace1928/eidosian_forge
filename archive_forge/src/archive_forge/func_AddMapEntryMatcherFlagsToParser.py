from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
def AddMapEntryMatcherFlagsToParser(parser):
    """Adds flags defining certificate map entry matcher."""
    is_primary_flag = base.Argument('--set-primary', help='The certificate will be used as the default cert if no other certificate in the map matches on SNI.', action='store_true')
    hostname_flag = base.Argument('--hostname', help='A domain name (FQDN), which controls when list of certificates specified in the resource will be taken under consideration for certificate selection.')
    group = base.ArgumentGroup(help='Arguments to configure matcher for the certificate map entry.', required=True, mutex=True, category=base.COMMONLY_USED_FLAGS)
    group.AddArgument(is_primary_flag)
    group.AddArgument(hostname_flag)
    group.AddToParser(parser)