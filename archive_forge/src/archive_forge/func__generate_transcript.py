import argparse
import cmd
import functools
import glob
import inspect
import os
import pydoc
import re
import sys
import threading
from code import (
from collections import (
from contextlib import (
from types import (
from typing import (
from . import (
from .argparse_custom import (
from .clipboard import (
from .command_definition import (
from .constants import (
from .decorators import (
from .exceptions import (
from .history import (
from .parsing import (
from .rl_utils import (
from .table_creator import (
from .utils import (
def _generate_transcript(self, history: Union[List[HistoryItem], List[str]], transcript_file: str) -> None:
    """Generate a transcript file from a given history of commands"""
    self.last_result = False
    transcript_path = os.path.abspath(os.path.expanduser(transcript_file))
    transcript_dir = os.path.dirname(transcript_path)
    if not os.path.isdir(transcript_dir) or not os.access(transcript_dir, os.W_OK):
        self.perror(f"'{transcript_dir}' is not a directory or you don't have write access")
        return
    commands_run = 0
    try:
        with self.sigint_protection:
            saved_echo = self.echo
            saved_stdout = self.stdout
            self.echo = False
        transcript = ''
        for history_item in history:
            first = True
            command = ''
            if isinstance(history_item, HistoryItem):
                history_item = history_item.raw
            for line in history_item.splitlines():
                if first:
                    command += f'{self.prompt}{line}\n'
                    first = False
                else:
                    command += f'{self.continuation_prompt}{line}\n'
            transcript += command
            stdsim = utils.StdSim(cast(TextIO, self.stdout))
            self.stdout = cast(TextIO, stdsim)
            try:
                stop = self.onecmd_plus_hooks(history_item, raise_keyboard_interrupt=True)
            except KeyboardInterrupt as ex:
                self.perror(ex)
                stop = True
            commands_run += 1
            transcript += stdsim.getvalue().replace('/', '\\/')
            if stop:
                break
    finally:
        with self.sigint_protection:
            self.echo = saved_echo
            self.stdout = cast(TextIO, saved_stdout)
    if commands_run < len(history):
        self.pwarning(f'Command {commands_run} triggered a stop and ended transcript generation early')
    try:
        with open(transcript_path, 'w') as fout:
            fout.write(transcript)
    except OSError as ex:
        self.perror(f"Error saving transcript file '{transcript_path}': {ex}")
    else:
        if commands_run == 1:
            plural = 'command and its output'
        else:
            plural = 'commands and their outputs'
        self.pfeedback(f"{commands_run} {plural} saved to transcript file '{transcript_path}'")
        self.last_result = True