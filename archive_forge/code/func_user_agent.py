import collections
import platform
import sys
def user_agent(name, version, extras=None):
    """Return an internet-friendly user_agent string.

    The majority of this code has been wilfully stolen from the equivalent
    function in Requests.

    :param name: The intended name of the user-agent, e.g. "python-requests".
    :param version: The version of the user-agent, e.g. "0.0.1".
    :param extras: List of two-item tuples that are added to the user-agent
        string.
    :returns: Formatted user-agent string
    :rtype: str
    """
    if extras is None:
        extras = []
    return UserAgentBuilder(name, version).include_extras(extras).include_implementation().include_system().build()