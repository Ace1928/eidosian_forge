from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
def _toggle_remove_duplicates(self):
    self._glue.remove_duplicates = not self._glue.remove_duplicates
    self._exampleList.selection_clear(0, 'end')
    self._readings = []
    self._populate_readingListbox()
    self._readingCache = [None for ex in self._examples]
    self._curExample = -1
    self._error = None
    self._drs = None
    self._redraw()