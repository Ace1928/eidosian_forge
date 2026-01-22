import sys
class CustomModuleType(StrEnum):
    SHA = 'securityHealthAnalyticsCustomModules'
    ETD = 'eventThreatDetectionCustomModules'
    EFFECTIVE_ETD = 'effectiveEventThreatDetectionCustomModules'
    EFFECTIVE_SHA = 'effectiveSecurityHealthAnalyticsCustomModules'