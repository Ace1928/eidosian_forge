import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
class LoadCorpus(threading.Thread):

    def __init__(self, name, model):
        threading.Thread.__init__(self)
        self.model, self.name = (model, name)

    def run(self):
        try:
            ts = self.model.CORPORA[self.name]()
            self.model.tagged_sents = [' '.join((w + '/' + t for w, t in sent)) for sent in ts]
            self.model.queue.put(CORPUS_LOADED_EVENT)
        except Exception as e:
            print(e)
            self.model.queue.put(ERROR_LOADING_CORPUS_EVENT)