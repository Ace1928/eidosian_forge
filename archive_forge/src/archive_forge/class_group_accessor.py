import logging
import re
class group_accessor:

    def __init__(self, m):
        self.match = m

    def group(self, i):
        try:
            return self.match.group(i) or 0
        except IndexError:
            return 0