import re
from ovs.db import error
def __raise_error(self, message):
    raise error.Error('Parsing %s failed: %s' % (self.name, message), self.json)