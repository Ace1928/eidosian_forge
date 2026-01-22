from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import operator
from googlecloudsdk.api_lib.kuberun import service
from googlecloudsdk.api_lib.kuberun import traffic
import six
def _ExtractTags(self):
    """Extracts the traffic tag state from spec and status into TrafficTags."""
    tags = {}
    for spec_target in self._spec_targets:
        if not spec_target.tag:
            continue
        tags[spec_target.tag] = TrafficTag(spec_target.tag, in_spec=True)
    for status_target in self._status_targets:
        if not status_target.tag:
            continue
        if status_target.tag in tags:
            tag = tags[status_target.tag]
        else:
            tag = tags.setdefault(status_target.tag, TrafficTag(status_target.tag))
        tag.url = status_target.url if status_target.url is not None else ''
        tag.inStatus = True
    self._tags = sorted(tags.values(), key=operator.attrgetter('tag'))