import re
from Bio import File
def _chop_end_misc(line):
    """Chops lines ending with  '     14-JUL-97  1CSA' and the like (PRIVATE)."""
    return re.sub('\\s+\\d\\d-\\w\\w\\w-\\d\\d\\s+[1-9][0-9A-Z]{3}\\s*\\Z', '', line)