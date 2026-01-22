from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import list_pager
from googlecloudsdk.api_lib.util import apis
def MatchesFilter(occurrence):
    if occurrence.kind != self.messages.Occurrence.KindValueValuesEnum.ATTESTATION:
        return False
    return True