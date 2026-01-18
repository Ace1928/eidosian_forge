import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def unfreeze_editable(self):
    self.query_box['state'] = 'normal'
    self.search_button['state'] = 'normal'
    self.set_paging_button_states()