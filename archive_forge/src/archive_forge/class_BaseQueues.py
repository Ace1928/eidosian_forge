from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from apitools.base.py import list_pager
from googlecloudsdk.core import exceptions
import six
class BaseQueues(object):
    """Client for queues service in the Cloud Tasks API."""

    def __init__(self, messages, queues_service):
        self.messages = messages
        self.queues_service = queues_service

    def Get(self, queue_ref):
        request = self.messages.CloudtasksProjectsLocationsQueuesGetRequest(name=queue_ref.RelativeName())
        return self.queues_service.Get(request)

    def List(self, parent_ref, limit=None, page_size=100):
        request = self.messages.CloudtasksProjectsLocationsQueuesListRequest(parent=parent_ref.RelativeName())
        return list_pager.YieldFromList(self.queues_service, request, batch_size=page_size, limit=limit, field='queues', batch_size_attribute='pageSize')

    def Delete(self, queue_ref):
        request = self.messages.CloudtasksProjectsLocationsQueuesDeleteRequest(name=queue_ref.RelativeName())
        return self.queues_service.Delete(request)

    def Purge(self, queue_ref):
        request = self.messages.CloudtasksProjectsLocationsQueuesPurgeRequest(name=queue_ref.RelativeName())
        return self.queues_service.Purge(request)

    def Pause(self, queue_ref):
        request = self.messages.CloudtasksProjectsLocationsQueuesPauseRequest(name=queue_ref.RelativeName())
        return self.queues_service.Pause(request)

    def Resume(self, queue_ref):
        request = self.messages.CloudtasksProjectsLocationsQueuesResumeRequest(name=queue_ref.RelativeName())
        return self.queues_service.Resume(request)

    def GetIamPolicy(self, queue_ref):
        request = self.messages.CloudtasksProjectsLocationsQueuesGetIamPolicyRequest(resource=queue_ref.RelativeName())
        return self.queues_service.GetIamPolicy(request)

    def SetIamPolicy(self, queue_ref, policy):
        request = self.messages.CloudtasksProjectsLocationsQueuesSetIamPolicyRequest(resource=queue_ref.RelativeName(), setIamPolicyRequest=self.messages.SetIamPolicyRequest(policy=policy))
        return self.queues_service.SetIamPolicy(request)