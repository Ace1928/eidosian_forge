from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AppEngineHttpQueue(_messages.Message):
    """App Engine HTTP queue. The task will be delivered to the App Engine
  application hostname specified by its AppEngineHttpQueue and
  AppEngineHttpRequest. The documentation for AppEngineHttpRequest explains
  how the task's host URL is constructed. Using AppEngineHttpQueue requires [`
  appengine.applications.get`](https://cloud.google.com/appengine/docs/admin-
  api/access-control) Google IAM permission for the project and the following
  scope: `https://www.googleapis.com/auth/cloud-platform`

  Fields:
    appEngineRoutingOverride: Overrides for the task-level app_engine_routing.
      If set, `app_engine_routing_override` is used for all tasks in the
      queue, no matter what the setting is for the task-level
      app_engine_routing.
  """
    appEngineRoutingOverride = _messages.MessageField('AppEngineRouting', 1)