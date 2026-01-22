from sentry_sdk import configure_scope
from sentry_sdk.hub import Hub
from sentry_sdk.integrations import Integration
from sentry_sdk.utils import capture_internal_exceptions
from sentry_sdk._types import TYPE_CHECKING
class SparkListener(object):

    def onApplicationEnd(self, applicationEnd):
        pass

    def onApplicationStart(self, applicationStart):
        pass

    def onBlockManagerAdded(self, blockManagerAdded):
        pass

    def onBlockManagerRemoved(self, blockManagerRemoved):
        pass

    def onBlockUpdated(self, blockUpdated):
        pass

    def onEnvironmentUpdate(self, environmentUpdate):
        pass

    def onExecutorAdded(self, executorAdded):
        pass

    def onExecutorBlacklisted(self, executorBlacklisted):
        pass

    def onExecutorBlacklistedForStage(self, executorBlacklistedForStage):
        pass

    def onExecutorMetricsUpdate(self, executorMetricsUpdate):
        pass

    def onExecutorRemoved(self, executorRemoved):
        pass

    def onJobEnd(self, jobEnd):
        pass

    def onJobStart(self, jobStart):
        pass

    def onNodeBlacklisted(self, nodeBlacklisted):
        pass

    def onNodeBlacklistedForStage(self, nodeBlacklistedForStage):
        pass

    def onNodeUnblacklisted(self, nodeUnblacklisted):
        pass

    def onOtherEvent(self, event):
        pass

    def onSpeculativeTaskSubmitted(self, speculativeTask):
        pass

    def onStageCompleted(self, stageCompleted):
        pass

    def onStageSubmitted(self, stageSubmitted):
        pass

    def onTaskEnd(self, taskEnd):
        pass

    def onTaskGettingResult(self, taskGettingResult):
        pass

    def onTaskStart(self, taskStart):
        pass

    def onUnpersistRDD(self, unpersistRDD):
        pass

    class Java:
        implements = ['org.apache.spark.scheduler.SparkListenerInterface']