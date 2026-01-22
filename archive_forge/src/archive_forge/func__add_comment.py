from __future__ import annotations
import os
import re
import shutil
import warnings
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING
import numpy as np
from monty.dev import deprecated
from monty.io import zopen
from monty.json import MSONable
from pymatgen.core import __version__ as CURRENT_VER
from pymatgen.io.core import InputFile
from pymatgen.io.lammps.data import CombinedData, LammpsData
from pymatgen.io.template import TemplateInputGen
def _add_comment(self, comment: str, inline: bool=False, stage_name: str | None=None, index_comment: bool=False) -> None:
    """
        Method to add a comment inside a stage (between actual commands)
        or as a whole stage (which will do nothing when LAMMPS runs).

        Args:
            comment (str): Comment string to be added. The comment will be
                preceded by '#' in the generated input.
            inline (bool): True if the comment should be inline within a given block of commands.
            stage_name (str): set the stage_name to which the comment will be written. Required if inline is True.
            index_comment (bool): True if the comment should start with "Comment x" with x its number in the ordering.
                Used only for inline comments.
        """
    if not inline:
        if stage_name is None:
            stage_name = f'Comment {self.ncomments + 1}'
            self.stages.append({'stage_name': stage_name, 'commands': [('#', comment)]})
        elif stage_name in self.stages_names:
            self._add_command(command='#', args=comment, stage_name=stage_name)
        else:
            self.stages.append({'stage_name': stage_name, 'commands': [('#', comment)]})
    elif stage_name:
        command = '#'
        if index_comment:
            if 'Comment' in comment and comment.strip()[9] == ':':
                args = ':'.join(comment.split(':')[1:])
            else:
                args = comment
        else:
            args = comment
        self._add_command(command=command, args=args, stage_name=stage_name)
    else:
        raise NotImplementedError('If you want to add an inline comment, please specify the stage name.')