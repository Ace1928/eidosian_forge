import abc
import collections
import collections.abc
import os
import sys
import typing
from typing import Optional, Dict, List
class PagerCommand(metaclass=abc.ABCMeta):
    """
    Abstract base class for pager commands.

    A subclass implementing this interface can be used to specify a particular
    pager command to run and its environment.
    """

    @abc.abstractmethod
    def command(self) -> List[str]:
        """Return the list of command arguments."""
        return ['more']

    @abc.abstractmethod
    def environment_variables(self, config: PagerConfig) -> Optional[Dict[str, str]]:
        """Return the dict of any environment variables to set."""
        return None