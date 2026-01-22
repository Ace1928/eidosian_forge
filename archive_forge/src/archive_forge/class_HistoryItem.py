import json
import re
from collections import (
from typing import (
import attr
from . import (
from .parsing import (
@attr.s(auto_attribs=True, frozen=True)
class HistoryItem:
    """Class used to represent one command in the history list"""
    _listformat = ' {:>4}  {}'
    _ex_listformat = ' {:>4}x {}'
    _statement_field = 'statement'
    statement: Statement = attr.ib(default=None, validator=attr.validators.instance_of(Statement))

    def __str__(self) -> str:
        """A convenient human readable representation of the history item"""
        return self.statement.raw

    @property
    def raw(self) -> str:
        """The raw input from the user for this item.

        Proxy property for ``self.statement.raw``
        """
        return self.statement.raw

    @property
    def expanded(self) -> str:
        """Return the command as run which includes shortcuts and aliases resolved
        plus any changes made in hooks

        Proxy property for ``self.statement.expanded_command_line``
        """
        return self.statement.expanded_command_line

    def pr(self, idx: int, script: bool=False, expanded: bool=False, verbose: bool=False) -> str:
        """Represent this item in a pretty fashion suitable for printing.

        If you pass verbose=True, script and expanded will be ignored

        :param idx: The 1-based index of this item in the history list
        :param script: True if formatting for a script (No item numbers)
        :param expanded: True if expanded command line should be printed
        :param verbose: True if expanded and raw should both appear when they are different
        :return: pretty print string version of a HistoryItem
        """
        if verbose:
            raw = self.raw.rstrip()
            expanded_command = self.expanded
            ret_str = self._listformat.format(idx, raw)
            if raw != expanded_command:
                ret_str += '\n' + self._ex_listformat.format(idx, expanded_command)
        else:
            if expanded:
                ret_str = self.expanded
            else:
                ret_str = self.raw.rstrip()
                if self.statement.multiline_command:
                    ret_str = ret_str.replace('\n', ' ')
            if not script:
                ret_str = self._listformat.format(idx, ret_str)
        return ret_str

    def to_dict(self) -> Dict[str, Any]:
        """Utility method to convert this HistoryItem into a dictionary for use in persistent JSON history files"""
        return {HistoryItem._statement_field: self.statement.to_dict()}

    @staticmethod
    def from_dict(source_dict: Dict[str, Any]) -> 'HistoryItem':
        """
        Utility method to restore a HistoryItem from a dictionary

        :param source_dict: source data dictionary (generated using to_dict())
        :return: HistoryItem object
        :raises KeyError: if source_dict is missing required elements
        """
        statement_dict = source_dict[HistoryItem._statement_field]
        return HistoryItem(Statement.from_dict(statement_dict))