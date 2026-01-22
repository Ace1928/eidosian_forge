from ... import errors, urlutils
from ...commands import Command
from ...option import Option
from ...trace import note
class cmd_gitlab_login(Command):
    __doc__ = "Log into a GitLab instance.\n\n    This command takes a GitLab instance URL (e.g. https://gitlab.com)\n    as well as an optional private token. Private tokens can be created via the\n    web UI.\n\n    :Examples:\n\n      Log into GNOME's GitLab (prompts for a token):\n\n         brz gitlab-login https://gitlab.gnome.org/\n\n      Log into Debian's salsa, using a token created earlier:\n\n         brz gitlab-login https://salsa.debian.org if4Theis6Eich7aef0zo\n    "
    takes_args = ['url', 'private_token?']
    takes_options = [Option('name', help='Name for GitLab site in configuration.', type=str), Option('no-check', "Don't check that the token is valid.")]

    def run(self, url, private_token=None, name=None, no_check=False):
        from breezy import ui
        from .forge import store_gitlab_token
        if name is None:
            try:
                name = urlutils.parse_url(url)[3].split('.')[-2]
            except (ValueError, IndexError):
                raise errors.CommandError('please specify a site name with --name')
        if private_token is None:
            note('Please visit %s to obtain a private token.', urlutils.join(url, '-/profile/personal_access_tokens'))
            private_token = ui.ui_factory.get_password('Private token')
        if not no_check:
            from breezy.transport import get_transport
            from .forge import GitLab
            GitLab(get_transport(url), private_token=private_token)
        store_gitlab_token(name=name, url=url, private_token=private_token)