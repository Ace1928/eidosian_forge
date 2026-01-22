from __future__ import annotations
import json
import os
from typing import Any
from pathlib import Path
from warnings import warn
from qiskit import user_config
class StyleDict(dict):
    """A dictionary for matplotlib styles.

    Defines additional abbreviations for key accesses, such as allowing
    ``"ec"`` instead of writing ``"edgecolor"``.
    """
    VALID_FIELDS = {'name', 'textcolor', 'gatetextcolor', 'subtextcolor', 'linecolor', 'creglinecolor', 'gatefacecolor', 'barrierfacecolor', 'backgroundcolor', 'edgecolor', 'fontsize', 'subfontsize', 'showindex', 'figwidth', 'dpi', 'margin', 'creglinestyle', 'displaytext', 'displaycolor'}
    ABBREVIATIONS = {'tc': 'textcolor', 'gt': 'gatetextcolor', 'sc': 'subtextcolor', 'lc': 'linecolor', 'cc': 'creglinecolor', 'gc': 'gatefacecolor', 'bc': 'barrierfacecolor', 'bg': 'backgroundcolor', 'ec': 'edgecolor', 'fs': 'fontsize', 'sfs': 'subfontsize', 'index': 'showindex', 'cline': 'creglinestyle', 'disptex': 'displaytext', 'dispcol': 'displaycolor'}

    def __setitem__(self, key: Any, value: Any) -> None:
        if key in self.ABBREVIATIONS.keys():
            key = self.ABBREVIATIONS[key]
        if key not in self.VALID_FIELDS:
            warn(f'style option ({key}) is not supported', UserWarning, 2)
        return super().__setitem__(key, value)

    def __getitem__(self, key: Any) -> Any:
        if key in self.ABBREVIATIONS.keys():
            key = self.ABBREVIATIONS[key]
        return super().__getitem__(key)

    def update(self, other):
        nested_attrs = {'displaycolor', 'displaytext'}
        for attr in nested_attrs.intersection(other.keys()):
            if attr in self.keys():
                self[attr].update(other[attr])
            else:
                self[attr] = other[attr]
        super().update(((key, value) for key, value in other.items() if key not in nested_attrs))