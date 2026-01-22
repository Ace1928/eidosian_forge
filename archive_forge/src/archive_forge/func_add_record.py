from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
from googlecloudsdk.api_lib.container.images import container_data_util
from googlecloudsdk.api_lib.containeranalysis import requests
import six
def add_record(self, occurrence):
    messages = requests.GetMessages()
    if occurrence.kind == messages.Occurrence.KindValueValuesEnum.VULNERABILITY:
        self.package_vulnerability_summary.add_record(occurrence)
    elif occurrence.kind == messages.Occurrence.KindValueValuesEnum.IMAGE:
        self.image_basis_summary.add_record(occurrence)
    elif occurrence.kind == messages.Occurrence.KindValueValuesEnum.BUILD:
        self.build_details_summary.add_record(occurrence)
    elif occurrence.kind == messages.Occurrence.KindValueValuesEnum.DEPLOYMENT:
        self.deployment_summary.add_record(occurrence)
    elif occurrence.kind == messages.Occurrence.KindValueValuesEnum.DISCOVERY:
        self.discovery_summary.add_record(occurrence)