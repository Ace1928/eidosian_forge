from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.components import completers
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.updater import update_manager
class Remove(base.SilentCommand):
    """Remove a registered Trusted Test component repository.
  """
    detailed_help = {'DESCRIPTION': '          Remove a registered Trusted Tester component repository from the list\n          of repositories used by the component manager.  After removing a\n          repository, you can run:\n            $ gcloud components update\n          to revert back to the standard version of any components that were\n          installed from that repository.\n      ', 'EXAMPLES': '          To be prompted for registered Trusted Tester component repositories to\n          remove run:\n\n            $ gcloud components repositories remove\n      '}

    @staticmethod
    def Args(parser):
        parser.add_argument('url', nargs='*', metavar='URL', completer=completers.RepoCompleter, help='Zero or more URLs for the component repositories you want to remove.  If none are given, you will be prompted to choose which existing repository you want to remove.')
        parser.add_argument('--all', action='store_true', help='Remove all registered repositories.')

    def Run(self, args):
        """Runs the remove command."""
        repos = update_manager.UpdateManager.GetAdditionalRepositories()
        removed_repos = []
        if args.all:
            removed_repos.extend(repos)
            repos = []
        elif args.url:
            if not repos:
                raise update_manager.NoRegisteredRepositoriesError('You have no registered repositories.')
            for url in args.url:
                if url not in repos:
                    raise exceptions.InvalidArgumentException('url', 'URL [{0}] was not a known registered repository.'.format(url))
            for url in args.url:
                repos.remove(url)
            removed_repos.extend(args.url)
        else:
            if not repos:
                raise update_manager.NoRegisteredRepositoriesError('You have no registered repositories.')
            result = console_io.PromptChoice(repos, default=None, message='Which repository would you like to remove?')
            if result is None:
                log.status.Print('No repository was removed.')
            else:
                removed_repos.append(repos.pop(result))
        if removed_repos:
            properties.PersistProperty(properties.VALUES.component_manager.additional_repositories, ','.join(repos) if repos else None, scope=properties.Scope.INSTALLATION)
        for removed_repo in removed_repos:
            log.status.Print('Removed repository: [{repo}]'.format(repo=removed_repo))
        return removed_repos