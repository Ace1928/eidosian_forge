from __future__ import absolute_import
from __future__ import print_function
import webbrowser
from ..palette import Palette
def wap(self):
    """
        View this color palette on the web.
        Will open a new tab in your web browser.

        """
    webbrowser.open_new_tab(self.url)