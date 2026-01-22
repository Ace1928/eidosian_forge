from __future__ import annotations
from .util import event_class, T_JSON_DICT
from dataclasses import dataclass
import enum
import typing
from . import dom
from . import network
from . import page
class PrerenderFinalStatus(enum.Enum):
    """
    List of FinalStatus reasons for Prerender2.
    """
    ACTIVATED = 'Activated'
    DESTROYED = 'Destroyed'
    LOW_END_DEVICE = 'LowEndDevice'
    INVALID_SCHEME_REDIRECT = 'InvalidSchemeRedirect'
    INVALID_SCHEME_NAVIGATION = 'InvalidSchemeNavigation'
    NAVIGATION_REQUEST_BLOCKED_BY_CSP = 'NavigationRequestBlockedByCsp'
    MAIN_FRAME_NAVIGATION = 'MainFrameNavigation'
    MOJO_BINDER_POLICY = 'MojoBinderPolicy'
    RENDERER_PROCESS_CRASHED = 'RendererProcessCrashed'
    RENDERER_PROCESS_KILLED = 'RendererProcessKilled'
    DOWNLOAD = 'Download'
    TRIGGER_DESTROYED = 'TriggerDestroyed'
    NAVIGATION_NOT_COMMITTED = 'NavigationNotCommitted'
    NAVIGATION_BAD_HTTP_STATUS = 'NavigationBadHttpStatus'
    CLIENT_CERT_REQUESTED = 'ClientCertRequested'
    NAVIGATION_REQUEST_NETWORK_ERROR = 'NavigationRequestNetworkError'
    CANCEL_ALL_HOSTS_FOR_TESTING = 'CancelAllHostsForTesting'
    DID_FAIL_LOAD = 'DidFailLoad'
    STOP = 'Stop'
    SSL_CERTIFICATE_ERROR = 'SslCertificateError'
    LOGIN_AUTH_REQUESTED = 'LoginAuthRequested'
    UA_CHANGE_REQUIRES_RELOAD = 'UaChangeRequiresReload'
    BLOCKED_BY_CLIENT = 'BlockedByClient'
    AUDIO_OUTPUT_DEVICE_REQUESTED = 'AudioOutputDeviceRequested'
    MIXED_CONTENT = 'MixedContent'
    TRIGGER_BACKGROUNDED = 'TriggerBackgrounded'
    MEMORY_LIMIT_EXCEEDED = 'MemoryLimitExceeded'
    DATA_SAVER_ENABLED = 'DataSaverEnabled'
    TRIGGER_URL_HAS_EFFECTIVE_URL = 'TriggerUrlHasEffectiveUrl'
    ACTIVATED_BEFORE_STARTED = 'ActivatedBeforeStarted'
    INACTIVE_PAGE_RESTRICTION = 'InactivePageRestriction'
    START_FAILED = 'StartFailed'
    TIMEOUT_BACKGROUNDED = 'TimeoutBackgrounded'
    CROSS_SITE_REDIRECT_IN_INITIAL_NAVIGATION = 'CrossSiteRedirectInInitialNavigation'
    CROSS_SITE_NAVIGATION_IN_INITIAL_NAVIGATION = 'CrossSiteNavigationInInitialNavigation'
    SAME_SITE_CROSS_ORIGIN_REDIRECT_NOT_OPT_IN_IN_INITIAL_NAVIGATION = 'SameSiteCrossOriginRedirectNotOptInInInitialNavigation'
    SAME_SITE_CROSS_ORIGIN_NAVIGATION_NOT_OPT_IN_IN_INITIAL_NAVIGATION = 'SameSiteCrossOriginNavigationNotOptInInInitialNavigation'
    ACTIVATION_NAVIGATION_PARAMETER_MISMATCH = 'ActivationNavigationParameterMismatch'
    ACTIVATED_IN_BACKGROUND = 'ActivatedInBackground'
    EMBEDDER_HOST_DISALLOWED = 'EmbedderHostDisallowed'
    ACTIVATION_NAVIGATION_DESTROYED_BEFORE_SUCCESS = 'ActivationNavigationDestroyedBeforeSuccess'
    TAB_CLOSED_BY_USER_GESTURE = 'TabClosedByUserGesture'
    TAB_CLOSED_WITHOUT_USER_GESTURE = 'TabClosedWithoutUserGesture'
    PRIMARY_MAIN_FRAME_RENDERER_PROCESS_CRASHED = 'PrimaryMainFrameRendererProcessCrashed'
    PRIMARY_MAIN_FRAME_RENDERER_PROCESS_KILLED = 'PrimaryMainFrameRendererProcessKilled'
    ACTIVATION_FRAME_POLICY_NOT_COMPATIBLE = 'ActivationFramePolicyNotCompatible'
    PRELOADING_DISABLED = 'PreloadingDisabled'
    BATTERY_SAVER_ENABLED = 'BatterySaverEnabled'
    ACTIVATED_DURING_MAIN_FRAME_NAVIGATION = 'ActivatedDuringMainFrameNavigation'
    PRELOADING_UNSUPPORTED_BY_WEB_CONTENTS = 'PreloadingUnsupportedByWebContents'
    CROSS_SITE_REDIRECT_IN_MAIN_FRAME_NAVIGATION = 'CrossSiteRedirectInMainFrameNavigation'
    CROSS_SITE_NAVIGATION_IN_MAIN_FRAME_NAVIGATION = 'CrossSiteNavigationInMainFrameNavigation'
    SAME_SITE_CROSS_ORIGIN_REDIRECT_NOT_OPT_IN_IN_MAIN_FRAME_NAVIGATION = 'SameSiteCrossOriginRedirectNotOptInInMainFrameNavigation'
    SAME_SITE_CROSS_ORIGIN_NAVIGATION_NOT_OPT_IN_IN_MAIN_FRAME_NAVIGATION = 'SameSiteCrossOriginNavigationNotOptInInMainFrameNavigation'
    MEMORY_PRESSURE_ON_TRIGGER = 'MemoryPressureOnTrigger'
    MEMORY_PRESSURE_AFTER_TRIGGERED = 'MemoryPressureAfterTriggered'
    PRERENDERING_DISABLED_BY_DEV_TOOLS = 'PrerenderingDisabledByDevTools'
    SPECULATION_RULE_REMOVED = 'SpeculationRuleRemoved'
    ACTIVATED_WITH_AUXILIARY_BROWSING_CONTEXTS = 'ActivatedWithAuxiliaryBrowsingContexts'
    MAX_NUM_OF_RUNNING_EAGER_PRERENDERS_EXCEEDED = 'MaxNumOfRunningEagerPrerendersExceeded'
    MAX_NUM_OF_RUNNING_NON_EAGER_PRERENDERS_EXCEEDED = 'MaxNumOfRunningNonEagerPrerendersExceeded'
    MAX_NUM_OF_RUNNING_EMBEDDER_PRERENDERS_EXCEEDED = 'MaxNumOfRunningEmbedderPrerendersExceeded'
    PRERENDERING_URL_HAS_EFFECTIVE_URL = 'PrerenderingUrlHasEffectiveUrl'
    REDIRECTED_PRERENDERING_URL_HAS_EFFECTIVE_URL = 'RedirectedPrerenderingUrlHasEffectiveUrl'
    ACTIVATION_URL_HAS_EFFECTIVE_URL = 'ActivationUrlHasEffectiveUrl'

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, json):
        return cls(json)