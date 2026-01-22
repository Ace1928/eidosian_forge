import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def _init_top(self, top):
    top.geometry('950x680+50+50')
    top.title('NLTK Concordance Search')
    top.bind('<Control-q>', self.destroy)
    top.protocol('WM_DELETE_WINDOW', self.destroy)
    top.minsize(950, 680)