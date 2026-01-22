import io
import re
from Bio.SeqFeature import SeqFeature, SimpleLocation, Position
class SwissProtParserError(ValueError):
    """An error occurred while parsing a SwissProt file."""

    def __init__(self, *args, line=None):
        """Create a SwissProtParserError object with the offending line."""
        super().__init__(*args)
        self.line = line