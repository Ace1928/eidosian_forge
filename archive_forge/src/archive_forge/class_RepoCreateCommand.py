import subprocess
from argparse import ArgumentParser
from typing import List, Union
from huggingface_hub.hf_api import HfFolder, create_repo, whoami
from requests.exceptions import HTTPError
from . import BaseTransformersCLICommand
class RepoCreateCommand(BaseUserCommand):

    def run(self):
        print(ANSI.red('WARNING! Managing repositories through transformers-cli is deprecated. Please use `huggingface-cli` instead.'))
        token = HfFolder.get_token()
        if token is None:
            print('Not logged in')
            exit(1)
        try:
            stdout = subprocess.check_output(['git', '--version']).decode('utf-8')
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print('Looks like you do not have git installed, please install.')
        try:
            stdout = subprocess.check_output(['git-lfs', '--version']).decode('utf-8')
            print(ANSI.gray(stdout.strip()))
        except FileNotFoundError:
            print(ANSI.red('Looks like you do not have git-lfs installed, please install. You can install from https://git-lfs.github.com/. Then run `git lfs install` (you only have to do this once).'))
        print('')
        user, _ = whoami(token)
        namespace = self.args.organization if self.args.organization is not None else user
        full_name = f'{namespace}/{self.args.name}'
        print(f'You are about to create {ANSI.bold(full_name)}')
        if not self.args.yes:
            choice = input('Proceed? [Y/n] ').lower()
            if not (choice == '' or choice == 'y' or choice == 'yes'):
                print('Abort')
                exit()
        try:
            url = create_repo(token, name=self.args.name, organization=self.args.organization)
        except HTTPError as e:
            print(e)
            print(ANSI.red(e.response.text))
            exit(1)
        print('\nYour repo now lives at:')
        print(f'  {ANSI.bold(url)}')
        print('\nYou can clone it locally with the command below, and commit/push as usual.')
        print(f'\n  git clone {url}')
        print('')