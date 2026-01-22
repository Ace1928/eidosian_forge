from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
class SentryListener(SparkListener):

    def __init__(self):
        self.hub = Hub.current

    def onJobStart(self, jobStart):
        message = 'Job {} Started'.format(jobStart.jobId())
        self.hub.add_breadcrumb(level='info', message=message)
        _set_app_properties()

    def onJobEnd(self, jobEnd):
        level = ''
        message = ''
        data = {'result': jobEnd.jobResult().toString()}
        if jobEnd.jobResult().toString() == 'JobSucceeded':
            level = 'info'
            message = 'Job {} Ended'.format(jobEnd.jobId())
        else:
            level = 'warning'
            message = 'Job {} Failed'.format(jobEnd.jobId())
        self.hub.add_breadcrumb(level=level, message=message, data=data)

    def onStageSubmitted(self, stageSubmitted):
        stage_info = stageSubmitted.stageInfo()
        message = 'Stage {} Submitted'.format(stage_info.stageId())
        data = {'attemptId': stage_info.attemptId(), 'name': stage_info.name()}
        self.hub.add_breadcrumb(level='info', message=message, data=data)
        _set_app_properties()

    def onStageCompleted(self, stageCompleted):
        from py4j.protocol import Py4JJavaError
        stage_info = stageCompleted.stageInfo()
        message = ''
        level = ''
        data = {'attemptId': stage_info.attemptId(), 'name': stage_info.name()}
        try:
            data['reason'] = stage_info.failureReason().get()
            message = 'Stage {} Failed'.format(stage_info.stageId())
            level = 'warning'
        except Py4JJavaError:
            message = 'Stage {} Completed'.format(stage_info.stageId())
            level = 'info'
        self.hub.add_breadcrumb(level=level, message=message, data=data)