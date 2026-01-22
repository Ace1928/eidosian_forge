import subprocess
from argparse import ArgumentParser
from typing import List, Union
from huggingface_hub.hf_api import HfFolder, create_repo, whoami
from requests.exceptions import HTTPError
from . import BaseTransformersCLICommand
class UserCommands(BaseTransformersCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        login_parser = parser.add_parser('login', help='Log in using the same credentials as on huggingface.co')
        login_parser.set_defaults(func=lambda args: LoginCommand(args))
        whoami_parser = parser.add_parser('whoami', help='Find out which huggingface.co account you are logged in as.')
        whoami_parser.set_defaults(func=lambda args: WhoamiCommand(args))
        logout_parser = parser.add_parser('logout', help='Log out')
        logout_parser.set_defaults(func=lambda args: LogoutCommand(args))
        repo_parser = parser.add_parser('repo', help='Deprecated: use `huggingface-cli` instead. Commands to interact with your huggingface.co repos.')
        repo_subparsers = repo_parser.add_subparsers(help='Deprecated: use `huggingface-cli` instead. huggingface.co repos related commands')
        repo_create_parser = repo_subparsers.add_parser('create', help='Deprecated: use `huggingface-cli` instead. Create a new repo on huggingface.co')
        repo_create_parser.add_argument('name', type=str, help="Name for your model's repo. Will be namespaced under your username to build the model id.")
        repo_create_parser.add_argument('--organization', type=str, help='Optional: organization namespace.')
        repo_create_parser.add_argument('-y', '--yes', action='store_true', help='Optional: answer Yes to the prompt')
        repo_create_parser.set_defaults(func=lambda args: RepoCreateCommand(args))