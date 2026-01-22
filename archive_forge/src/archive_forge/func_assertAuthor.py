import os
import re
from io import BytesIO, StringIO
from .. import (branchbuilder, errors, gpg, log, registry, revision,
def assertAuthor(expected, author):
    self.rev.properties['author'] = author
    self.assertEqual(expected, self.lf.short_author(self.rev))