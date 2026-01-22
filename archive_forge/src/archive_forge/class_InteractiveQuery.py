import threading
from paramiko import util
from paramiko.common import (
class InteractiveQuery:
    """
    A query (set of prompts) for a user during interactive authentication.
    """

    def __init__(self, name='', instructions='', *prompts):
        """
        Create a new interactive query to send to the client.  The name and
        instructions are optional, but are generally displayed to the end
        user.  A list of prompts may be included, or they may be added via
        the `add_prompt` method.

        :param str name: name of this query
        :param str instructions:
            user instructions (usually short) about this query
        :param str prompts: one or more authentication prompts
        """
        self.name = name
        self.instructions = instructions
        self.prompts = []
        for x in prompts:
            if isinstance(x, str):
                self.add_prompt(x)
            else:
                self.add_prompt(x[0], x[1])

    def add_prompt(self, prompt, echo=True):
        """
        Add a prompt to this query.  The prompt should be a (reasonably short)
        string.  Multiple prompts can be added to the same query.

        :param str prompt: the user prompt
        :param bool echo:
            ``True`` (default) if the user's response should be echoed;
            ``False`` if not (for a password or similar)
        """
        self.prompts.append((prompt, echo))