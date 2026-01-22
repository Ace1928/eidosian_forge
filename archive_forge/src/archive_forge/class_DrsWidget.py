from nltk.parse import MaltParser
from nltk.sem.drt import DrsDrawer, DrtVariableExpression
from nltk.sem.glue import DrtGlue
from nltk.sem.logic import Variable
from nltk.tag import RegexpTagger
from nltk.util import in_idle
class DrsWidget:

    def __init__(self, canvas, drs, **attribs):
        self._drs = drs
        self._canvas = canvas
        canvas.font = Font(font=canvas.itemcget(canvas.create_text(0, 0, text=''), 'font'))
        canvas._BUFFER = 3
        self.bbox = (0, 0, 0, 0)

    def draw(self):
        right, bottom = DrsDrawer(self._drs, canvas=self._canvas).draw()
        self.bbox = (0, 0, right + 1, bottom + 1)

    def clear(self):
        self._canvas.create_rectangle(self.bbox, fill='white', width='0')