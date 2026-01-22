import importlib.resources as pkg_resources
import logging
from pathlib import Path
from typing import Any, List, Tuple
import yaml
from . import resources
from .deprecation_utils import deprecated
@deprecated('Use `huggingface_hub.DatasetCard` instead.')
class ReadMe(Section):

    def __init__(self, name: str, lines: List[str], structure: dict=None, suppress_parsing_errors: bool=False):
        super().__init__(name=name, level='')
        self.structure = structure
        self.yaml_tags_line_count = -2
        self.tag_count = 0
        self.lines = lines
        if self.lines is not None:
            self.parse(suppress_parsing_errors=suppress_parsing_errors)

    def validate(self):
        if self.structure is None:
            content, error_list, warning_list = self._validate(readme_structure)
        else:
            content, error_list, warning_list = self._validate(self.structure)
        if error_list != [] or warning_list != []:
            errors = '\n'.join(['-\t' + x for x in error_list + warning_list])
            error_string = f'The following issues were found for the README at `{self.name}`:\n' + errors
            raise ValueError(error_string)

    @classmethod
    def from_readme(cls, path: Path, structure: dict=None, suppress_parsing_errors: bool=False):
        with open(path, encoding='utf-8') as f:
            lines = f.readlines()
        return cls(path, lines, structure, suppress_parsing_errors=suppress_parsing_errors)

    @classmethod
    def from_string(cls, string: str, structure: dict=None, root_name: str='root', suppress_parsing_errors: bool=False):
        lines = string.split('\n')
        return cls(root_name, lines, structure, suppress_parsing_errors=suppress_parsing_errors)

    def parse(self, suppress_parsing_errors: bool=False):
        line_count = 0
        for line in self.lines:
            self.yaml_tags_line_count += 1
            if line.strip(' \n') == '---':
                self.tag_count += 1
                if self.tag_count == 2:
                    break
            line_count += 1
        if self.tag_count == 2:
            self.lines = self.lines[line_count + 1:]
        else:
            self.lines = self.lines[self.tag_count:]
        super().parse(suppress_parsing_errors=suppress_parsing_errors)

    def __str__(self):
        """Returns the string of dictionary representation of the ReadMe."""
        return str(self.to_dict())

    def _validate(self, readme_structure):
        error_list = []
        warning_list = []
        if self.yaml_tags_line_count == 0:
            warning_list.append('Empty YAML markers are present in the README.')
        elif self.tag_count == 0:
            warning_list.append('No YAML markers are present in the README.')
        elif self.tag_count == 1:
            warning_list.append('Only the start of YAML tags present in the README.')
        num_first_level_keys = len(self.content.keys())
        if num_first_level_keys > 1:
            error_list.append(f'The README has several first-level headings: {', '.join(['`' + x + '`' for x in list(self.content.keys())])}. Only one heading is expected. Skipping further validation for this README.')
        elif num_first_level_keys < 1:
            error_list.append('The README has no first-level headings. One heading is expected. Skipping further validation for this README.')
        else:
            start_key = list(self.content.keys())[0]
            if start_key.startswith('Dataset Card for'):
                _, sec_error_list, sec_warning_list = self.content[start_key].validate(readme_structure['subsections'][0])
                error_list += sec_error_list
                warning_list += sec_warning_list
            else:
                error_list.append('No first-level heading starting with `Dataset Card for` found in README. Skipping further validation for this README.')
        if error_list:
            return ({}, error_list, warning_list)
        else:
            return (self.to_dict(), error_list, warning_list)