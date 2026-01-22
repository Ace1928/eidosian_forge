from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import arg_parsers
def GetHasPrivateIssuer(args):
    return getattr(args, 'has_private_issuer', None)