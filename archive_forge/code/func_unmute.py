import sys
import logging
def unmute(self):
    """
        Resume logging to stdout.
        """
    self.streamHandler.level = self.prev_level