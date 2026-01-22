import configobj
from . import bedding, cmdline, errors, globbing, osutils
class _RulesSearcher:
    """An object that provides rule-based preferences."""

    def get_items(self, path):
        """Return the preferences for a path as name,value tuples.

        :param path: tree relative path
        :return: () if no rule matched, otherwise a sequence of name,value
          tuples.
        """
        raise NotImplementedError(self.get_items)

    def get_selected_items(self, path, names):
        """Return selected preferences for a path as name,value tuples.

        :param path: tree relative path
        :param names: the list of preferences to lookup
        :return: () if no rule matched, otherwise a sequence of name,value
          tuples. The sequence is the same length as names,
          tuple order matches the order in names, and
          undefined preferences are given the value None.
        """
        raise NotImplementedError(self.get_selected_items)

    def get_single_value(self, path, preference_name):
        """Get a single preference for a single file.

        :returns: The string preference value, or None.
        """
        for key, value in self.get_selected_items(path, [preference_name]):
            return value
        return None