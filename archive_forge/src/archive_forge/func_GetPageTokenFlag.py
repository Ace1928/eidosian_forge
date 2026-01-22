from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
def GetPageTokenFlag():
    return base.Argument('--page_token', help='Page token received from a previous call. Provide this token to retrieve the next page.')