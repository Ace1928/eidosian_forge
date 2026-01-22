import inspect
import logging
import stevedore
class EntryPointWrapper(object):
    """Wrap up a command class already imported to make it look like a plugin.
    """

    def __init__(self, name, command_class):
        self.name = name
        self.command_class = command_class

    def load(self, require=False):
        return self.command_class