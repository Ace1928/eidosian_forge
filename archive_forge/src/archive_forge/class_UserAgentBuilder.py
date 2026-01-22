import collections
import platform
import sys
class UserAgentBuilder(object):
    """Class to provide a greater level of control than :func:`user_agent`.

    This is used by :func:`user_agent` to build its User-Agent string.

    .. code-block:: python

        user_agent_str = UserAgentBuilder(
                name='requests-toolbelt',
                version='17.4.0',
            ).include_implementation(
            ).include_system(
            ).include_extras([
                ('requests', '2.14.2'),
                ('urllib3', '1.21.2'),
            ]).build()

    """
    format_string = '%s/%s'

    def __init__(self, name, version):
        """Initialize our builder with the name and version of our user agent.

        :param str name:
            Name of our user-agent.
        :param str version:
            The version string for user-agent.
        """
        self._pieces = collections.deque([(name, version)])

    def build(self):
        """Finalize the User-Agent string.

        :returns:
            Formatted User-Agent string.
        :rtype:
            str
        """
        return ' '.join([self.format_string % piece for piece in self._pieces])

    def include_extras(self, extras):
        """Include extra portions of the User-Agent.

        :param list extras:
            list of tuples of extra-name and extra-version
        """
        if any((len(extra) != 2 for extra in extras)):
            raise ValueError('Extras should be a sequence of two item tuples.')
        self._pieces.extend(extras)
        return self

    def include_implementation(self):
        """Append the implementation string to the user-agent string.

        This adds the the information that you're using CPython 2.7.13 to the
        User-Agent.
        """
        self._pieces.append(_implementation_tuple())
        return self

    def include_system(self):
        """Append the information about the Operating System."""
        self._pieces.append(_platform_tuple())
        return self