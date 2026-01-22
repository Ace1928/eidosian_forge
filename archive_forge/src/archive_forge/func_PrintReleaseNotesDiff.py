from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.core import config
from googlecloudsdk.core import log
from googlecloudsdk.core.document_renderers import render_document
from googlecloudsdk.core.updater import installers
import requests
from six.moves import StringIO
def PrintReleaseNotesDiff(release_notes_url, current_version, latest_version):
    """Prints the release notes diff based on your current version.

  If any of the arguments are None, a generic message will be printed telling
  the user to go to the web to view the release notes.  If the release_notes_url
  is also None, it will print the developers site page for the SDK.

  Args:
    release_notes_url: str, The URL to download the latest release notes from.
    current_version: str, The current version of the SDK you have installed.
    latest_version: str, The version you are about to update to.
  """
    if release_notes_url and current_version and latest_version:
        notes = ReleaseNotes.FromURL(release_notes_url)
        if notes:
            release_notes_diff = notes.Diff(latest_version, current_version)
        else:
            release_notes_diff = None
    else:
        release_notes_diff = None
    if not release_notes_diff:
        log.status.write('For the latest full release notes, please visit:\n  {0}\n\n'.format(config.INSTALLATION_CONFIG.release_notes_url))
        return
    if len(release_notes_diff) > ReleaseNotes.MAX_DIFF:
        log.status.Print('A lot has changed since your last upgrade.  For the latest full release notes,\nplease visit:\n  {0}\n'.format(config.INSTALLATION_CONFIG.release_notes_url))
        return
    log.status.Print('The following release notes are new in this upgrade.\nPlease read carefully for information about new features, breaking changes,\nand bugs fixed.  The latest full release notes can be viewed at:\n  {0}\n'.format(config.INSTALLATION_CONFIG.release_notes_url))
    full_text = StringIO()
    for _, text in release_notes_diff:
        full_text.write(text)
        full_text.write('\n')
    full_text.seek(0)
    render_document.RenderDocument('text', full_text, log.status)
    log.status.Print()