from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.util import apis as core_apis
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
def _FormatVersion(self, service_name, version_id):
    res = resources.REGISTRY.Parse(version_id, params={'appsId': self.project, 'servicesId': service_name}, collection='appengine.apps.services.versions')
    return res.RelativeName()