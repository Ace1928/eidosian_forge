from __future__ import annotations
import logging
import os
import sys
import typing as t
from copy import deepcopy
from pathlib import Path
from shutil import which
from traitlets import Bool, List, Unicode, observe
from traitlets.config.application import Application, catch_config_error
from traitlets.config.loader import ConfigFileNotFound
from .paths import (
from .utils import ensure_dir_exists, ensure_event_loop
def write_default_config(self) -> None:
    """Write our default config to a .py config file"""
    if self.config_file:
        config_file = self.config_file
    else:
        config_file = str(Path(self.config_dir, self.config_file_name + '.py'))
    if Path(config_file).exists() and (not self.answer_yes):
        answer = ''

        def ask() -> str:
            prompt = 'Overwrite %s with default config? [y/N]' % config_file
            try:
                return input(prompt).lower() or 'n'
            except KeyboardInterrupt:
                print('')
                return 'n'
        answer = ask()
        while not answer.startswith(('y', 'n')):
            print("Please answer 'yes' or 'no'")
            answer = ask()
        if answer.startswith('n'):
            return
    config_text = self.generate_config_file()
    print('Writing default config to: %s' % config_file)
    ensure_dir_exists(Path(config_file).parent.resolve(), 448)
    with Path.open(Path(config_file), mode='w', encoding='utf-8') as f:
        f.write(config_text)