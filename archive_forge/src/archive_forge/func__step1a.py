import re
from nltk.stem.api import StemmerI
def _step1a(self, word):
    """Implements Step 1a from "An algorithm for suffix stripping"

        From the paper:

            SSES -> SS                         caresses  ->  caress
            IES  -> I                          ponies    ->  poni
                                               ties      ->  ti
            SS   -> SS                         caress    ->  caress
            S    ->                            cats      ->  cat
        """
    if self.mode == self.NLTK_EXTENSIONS:
        if word.endswith('ies') and len(word) == 4:
            return self._replace_suffix(word, 'ies', 'ie')
    return self._apply_rule_list(word, [('sses', 'ss', None), ('ies', 'i', None), ('ss', 'ss', None), ('s', '', None)])