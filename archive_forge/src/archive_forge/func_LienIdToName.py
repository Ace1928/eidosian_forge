from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.core import resources
def LienIdToName(lien_id):
    return 'liens/{0}'.format(lien_id)