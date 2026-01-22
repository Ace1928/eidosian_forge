import bisect
import textwrap
from collections import defaultdict
from nltk.tag import BrillTagger, untag
def _find_rules(self, sent, wordnum, new_tag):
    """
        Use the templates to find rules that apply at index *wordnum*
        in the sentence *sent* and generate the tag *new_tag*.
        """
    for template in self._templates:
        yield from template.applicable_rules(sent, wordnum, new_tag)