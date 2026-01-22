import codecs
import os
import sys
from io import BytesIO, StringIO
from subprocess import call
from . import bedding, cmdline, config, osutils, trace, transport, ui
from .errors import BzrError
from .hooks import Hooks
class MessageEditorHooks(Hooks):
    """A dictionary mapping hook name to a list of callables for message editor
    hooks.

    e.g. ['commit_message_template'] is the list of items to be called to
    generate a commit message template
    """

    def __init__(self):
        """Create the default hooks.

        These are all empty initially.
        """
        Hooks.__init__(self, 'breezy.msgeditor', 'hooks')
        self.add_hook('set_commit_message', 'Set a fixed commit message. set_commit_message is called with the breezy.commit.Commit object (so you can also change e.g. revision properties by editing commit.builder._revprops) and the message so far. set_commit_message must return the message to use or None if it should use the message editor as normal.', (2, 4))
        self.add_hook('commit_message_template', 'Called when a commit message is being generated. commit_message_template is called with the breezy.commit.Commit object and the message that is known so far. commit_message_template must return a new message to use (which could be the same as it was given). When there are multiple hooks registered for commit_message_template, they are chained with the result from the first passed into the second, and so on.', (1, 10))