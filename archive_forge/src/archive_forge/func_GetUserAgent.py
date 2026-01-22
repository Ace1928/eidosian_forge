import six
import sys
import gslib
from gslib.utils import system_util
from gslib.storage_url import StorageUrlFromString
from gslib.exception import InvalidUrlError
def GetUserAgent(args, metrics_off=True):
    """Using the command arguments return a suffix for the UserAgent string.

  Args:
    args: str[], parsed set of arguments entered in the CLI.
    metrics_off: boolean, whether the MetricsCollector is disabled.

  Returns:
    str, A string value that can be appended to an existing UserAgent.
  """
    user_agent = ' gsutil/%s' % gslib.VERSION
    user_agent += ' (%s)' % sys.platform
    user_agent += ' analytics/%s' % ('disabled' if metrics_off else 'enabled')
    user_agent += ' interactive/%s' % system_util.IsRunningInteractively()
    if len(args) > 0:
        user_agent += ' command/%s' % args[0]
        if len(args) > 2:
            if args[0] in ['cp', 'mv', 'rsync']:
                try:
                    src = StorageUrlFromString(six.ensure_text(args[-2]))
                    dst = StorageUrlFromString(six.ensure_text(args[-1]))
                    if src.IsCloudUrl() and dst.IsCloudUrl() and (src.scheme != dst.scheme):
                        user_agent += '-DaisyChain'
                except InvalidUrlError:
                    pass
            elif args[0] == 'rewrite':
                if '-k' in args:
                    user_agent += '-k'
                if '-s' in args:
                    user_agent += '-s'
    if system_util.InvokedViaCloudSdk():
        user_agent += ' google-cloud-sdk'
        if system_util.CloudSdkVersion():
            user_agent += '/%s' % system_util.CloudSdkVersion()
    return user_agent