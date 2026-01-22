from . import commands as _mod_commands
from . import errors, help_topics, osutils, plugin, ui, utextwrap
class HelpIndices:
    """Maintainer of help topics across multiple indices.

    It is currently separate to the HelpTopicRegistry because of its ordered
    nature, but possibly we should instead structure it as a search within the
    registry and add ordering and searching facilities to the registry. The
    registry would probably need to be restructured to support that cleanly
    which is why this has been implemented in parallel even though it does as a
    result permit searching for help in indices which are not discoverable via
    'help topics'.

    Each index has a unique prefix string, such as "commands", and contains
    help topics which can be listed or searched.
    """

    def __init__(self):
        self.search_path = [help_topics.HelpTopicIndex(), _mod_commands.HelpCommandIndex(), plugin.PluginsHelpIndex(), help_topics.ConfigOptionHelpIndex()]

    def _check_prefix_uniqueness(self):
        """Ensure that the index collection is able to differentiate safely."""
        prefixes = set()
        for index in self.search_path:
            prefix = index.prefix
            if prefix in prefixes:
                raise errors.DuplicateHelpPrefix(prefix)
            prefixes.add(prefix)

    def search(self, topic):
        """Search for topic across the help search path.

        :param topic: A string naming the help topic to search for.
        :raises: NoHelpTopic if none of the indexs in search_path have topic.
        :return: A list of HelpTopics which matched 'topic'.
        """
        self._check_prefix_uniqueness()
        result = []
        for index in self.search_path:
            result.extend([(index, _topic) for _topic in index.get_topics(topic)])
        if not result:
            raise NoHelpTopic(topic)
        else:
            return result