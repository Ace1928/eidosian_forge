from typing import List, Tuple
from ...pipeline import Lemmatizer
from ...tokens import Token

    French language lemmatizer applies the default rule based lemmatization
    procedure with some modifications for better French language support.

    The parts of speech 'ADV', 'PRON', 'DET', 'ADP' and 'AUX' are added to use
    the rule-based lemmatization. As a last resort, the lemmatizer checks in
    the lookup table.
    