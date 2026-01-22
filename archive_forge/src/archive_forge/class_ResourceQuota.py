from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourceQuota(_messages.Message):
    """ResourceQuota is a subresource of Namespace, representing quotas that
  can be applied to all instances of the Namespace in relevant clusters.

  Fields:
    createTime: Output only. When the resource was created.
    deleteTime: Output only. When the resource was deleted.
    limitsCpu: A string attribute.
    limitsMemory: A string attribute.
    name: The resource name for the resourcequota itself `projects/{project}/l
      ocations/{location}/scopes/{scope}/namespaces/{namespace}/resourcequotas
      /{resourcequota}`
    requestsCpu: https://kubernetes.io/docs/concepts/policy/resource-
      quotas/#compute-resource-quota
    requestsMemory: A string attribute.
    state: Output only. State of the resource.
    uid: Output only. Google-generated UUID for this resource.
    updateTime: Output only. When the resource was last updated.
  """
    createTime = _messages.StringField(1)
    deleteTime = _messages.StringField(2)
    limitsCpu = _messages.StringField(3)
    limitsMemory = _messages.StringField(4)
    name = _messages.StringField(5)
    requestsCpu = _messages.StringField(6)
    requestsMemory = _messages.StringField(7)
    state = _messages.MessageField('ResourceQuotaLifecycleState', 8)
    uid = _messages.StringField(9)
    updateTime = _messages.StringField(10)