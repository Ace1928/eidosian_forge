from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core.updater import update_manager
def RepoCompleter(prefix, **unused_kwargs):
    """An argcomplete completer for currently added component repositories."""
    repos = update_manager.UpdateManager.GetAdditionalRepositories()
    return [r for r in repos if r.startswith(prefix)]