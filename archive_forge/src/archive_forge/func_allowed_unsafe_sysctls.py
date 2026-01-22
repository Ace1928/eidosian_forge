from pprint import pformat
from six import iteritems
import re
@allowed_unsafe_sysctls.setter
def allowed_unsafe_sysctls(self, allowed_unsafe_sysctls):
    """
        Sets the allowed_unsafe_sysctls of this
        PolicyV1beta1PodSecurityPolicySpec.
        allowedUnsafeSysctls is a list of explicitly allowed unsafe sysctls,
        defaults to none. Each entry is either a plain sysctl name or ends in
        "*" in which case it is considered as a prefix of allowed sysctls.
        Single * means all unsafe sysctls are allowed. Kubelet has to whitelist
        all allowed unsafe sysctls explicitly to avoid rejection.  Examples:
        e.g. "foo/*" allows "foo/bar", "foo/baz", etc. e.g. "foo.*"
        allows "foo.bar", "foo.baz", etc.

        :param allowed_unsafe_sysctls: The allowed_unsafe_sysctls of this
        PolicyV1beta1PodSecurityPolicySpec.
        :type: list[str]
        """
    self._allowed_unsafe_sysctls = allowed_unsafe_sysctls