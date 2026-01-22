from __future__ import unicode_literals
from abc import ABCMeta, abstractmethod
from six import with_metaclass
from .filters import to_cli_filter
class ConditionalAutoSuggest(AutoSuggest):
    """
    Auto suggest that can be turned on and of according to a certain condition.
    """

    def __init__(self, auto_suggest, filter):
        assert isinstance(auto_suggest, AutoSuggest)
        self.auto_suggest = auto_suggest
        self.filter = to_cli_filter(filter)

    def get_suggestion(self, cli, buffer, document):
        if self.filter(cli):
            return self.auto_suggest.get_suggestion(cli, buffer, document)