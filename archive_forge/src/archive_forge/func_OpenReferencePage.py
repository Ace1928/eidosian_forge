from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import os
import subprocess
import sys
import webbrowser
from googlecloudsdk.command_lib.interactive import parser
from googlecloudsdk.core import log
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def OpenReferencePage(cli, line, pos):
    """Opens a web browser or local help/man page for line at pos."""
    man_page = bool(encoding.GetEncodedValue(os.environ, 'SSH_CLIENT'))
    ref = _GetReferenceURL(cli, line, pos, man_page)
    if not ref:
        return
    if man_page:
        cli.Run(ref, alternate_screen=True)
        return
    webbrowser.subprocess = FakeSubprocessModule()
    try:
        browser = webbrowser.get()
        browser.open_new_tab(ref)
    except webbrowser.Error as e:
        cli.run_in_terminal(lambda: log.error('failed to open browser: %s', e))