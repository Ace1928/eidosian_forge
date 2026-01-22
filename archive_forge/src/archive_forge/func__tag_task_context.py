from __future__ import absolute_import
import sys
from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import (
from sentry_sdk._types import TYPE_CHECKING
def _tag_task_context():
    from pyspark.taskcontext import TaskContext
    with configure_scope() as scope:

        @scope.add_event_processor
        def process_event(event, hint):
            with capture_internal_exceptions():
                integration = Hub.current.get_integration(SparkWorkerIntegration)
                task_context = TaskContext.get()
                if integration is None or task_context is None:
                    return event
                event.setdefault('tags', {}).setdefault('stageId', str(task_context.stageId()))
                event['tags'].setdefault('partitionId', str(task_context.partitionId()))
                event['tags'].setdefault('attemptNumber', str(task_context.attemptNumber()))
                event['tags'].setdefault('taskAttemptId', str(task_context.taskAttemptId()))
                if task_context._localProperties:
                    if 'sentry_app_name' in task_context._localProperties:
                        event['tags'].setdefault('app_name', task_context._localProperties['sentry_app_name'])
                        event['tags'].setdefault('application_id', task_context._localProperties['sentry_application_id'])
                    if 'callSite.short' in task_context._localProperties:
                        event.setdefault('extra', {}).setdefault('callSite', task_context._localProperties['callSite.short'])
            return event