import re
from .api import FancyValidator
from .compound import Any
from .validators import Regex, Invalid, _
def assembly_formatstring(self, partition_lengths, delimiter):
    if len(partition_lengths) == 1:
        return _('%d digits') % partition_lengths[0]
    return delimiter.join(('n' * n for n in partition_lengths))