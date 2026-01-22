import logging
import stevedore
from cliff import command
class CompleteNoCode(CompleteShellBase):
    """completion with no code
    """

    def __init__(self, name, output):
        super(CompleteNoCode, self).__init__(name, output)

    def get_header(self):
        return ''

    def get_trailer(self):
        return ''