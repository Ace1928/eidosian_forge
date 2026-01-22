from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import platforms
def ShouldLaunchBrowser(attempt_launch_browser):
    """Determines if a browser can be launched.

  Args:
    attempt_launch_browser: bool, True to launch a browser if it's possible in
      the user's environment; False to not even try.

  Returns:
    True if the tool should actually launch a browser, based on user preference
    and environment.
  """
    import webbrowser
    launch_browser = attempt_launch_browser
    if launch_browser:
        current_os = platforms.OperatingSystem.Current()
        if current_os is platforms.OperatingSystem.LINUX and (not any((encoding.GetEncodedValue(os.environ, var) for var in _DISPLAY_VARIABLES))):
            launch_browser = False
        try:
            browser = webbrowser.get()
            if hasattr(browser, 'name') and browser.name in _WEBBROWSER_NAMES_BLOCKLIST:
                launch_browser = False
        except webbrowser.Error:
            launch_browser = False
    return launch_browser