import difflib
import json
import os
import re
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from datetime import date
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Pattern, Tuple, Union
import yaml
from ..models import auto as auto_module
from ..models.auto.configuration_auto import model_type_to_module_name
from ..utils import is_flax_available, is_tf_available, is_torch_available, logging
from . import BaseTransformersCLICommand
class AddNewModelLikeCommand(BaseTransformersCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        add_new_model_like_parser = parser.add_parser('add-new-model-like')
        add_new_model_like_parser.add_argument('--config_file', type=str, help='A file with all the information for this model creation.')
        add_new_model_like_parser.add_argument('--path_to_repo', type=str, help='When not using an editable install, the path to the Transformers repo.')
        add_new_model_like_parser.set_defaults(func=add_new_model_like_command_factory)

    def __init__(self, config_file=None, path_to_repo=None, *args):
        if config_file is not None:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            self.old_model_type = config['old_model_type']
            self.model_patterns = ModelPatterns(**config['new_model_patterns'])
            self.add_copied_from = config.get('add_copied_from', True)
            self.frameworks = config.get('frameworks', get_default_frameworks())
            self.old_checkpoint = config.get('old_checkpoint', None)
        else:
            self.old_model_type, self.model_patterns, self.add_copied_from, self.frameworks, self.old_checkpoint = get_user_input()
        self.path_to_repo = path_to_repo

    def run(self):
        if self.path_to_repo is not None:
            global TRANSFORMERS_PATH
            global REPO_PATH
            REPO_PATH = Path(self.path_to_repo)
            TRANSFORMERS_PATH = REPO_PATH / 'src' / 'transformers'
        create_new_model_like(model_type=self.old_model_type, new_model_patterns=self.model_patterns, add_copied_from=self.add_copied_from, frameworks=self.frameworks, old_checkpoint=self.old_checkpoint)