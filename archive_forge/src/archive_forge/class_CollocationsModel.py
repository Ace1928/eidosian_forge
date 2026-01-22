import queue as q
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.probability import FreqDist
from nltk.util import in_idle
class CollocationsModel:

    def __init__(self, queue):
        self.result_count = None
        self.selected_corpus = None
        self.collocations = None
        self.CORPORA = _CORPORA
        self.DEFAULT_CORPUS = _DEFAULT
        self.queue = queue
        self.reset_results()

    def reset_results(self):
        self.result_pages = []
        self.results_returned = 0

    def load_corpus(self, name):
        self.selected_corpus = name
        self.collocations = None
        runner_thread = self.LoadCorpus(name, self)
        runner_thread.start()
        self.reset_results()

    def non_default_corpora(self):
        copy = []
        copy.extend(list(self.CORPORA.keys()))
        copy.remove(self.DEFAULT_CORPUS)
        copy.sort()
        return copy

    def is_last_page(self, number):
        if number < len(self.result_pages):
            return False
        return self.results_returned + (number - len(self.result_pages)) * self.result_count >= len(self.collocations)

    def next(self, page):
        if len(self.result_pages) - 1 < page:
            for i in range(page - (len(self.result_pages) - 1)):
                self.result_pages.append(self.collocations[self.results_returned:self.results_returned + self.result_count])
                self.results_returned += self.result_count
        return self.result_pages[page]

    def prev(self, page):
        if page == -1:
            return []
        return self.result_pages[page]

    class LoadCorpus(threading.Thread):

        def __init__(self, name, model):
            threading.Thread.__init__(self)
            self.model, self.name = (model, name)

        def run(self):
            try:
                words = self.model.CORPORA[self.name]()
                from operator import itemgetter
                text = [w for w in words if len(w) > 2]
                fd = FreqDist((tuple(text[i:i + 2]) for i in range(len(text) - 1)))
                vocab = FreqDist(text)
                scored = [((w1, w2), fd[w1, w2] ** 3 / (vocab[w1] * vocab[w2])) for w1, w2 in fd]
                scored.sort(key=itemgetter(1), reverse=True)
                self.model.collocations = list(map(itemgetter(0), scored))
                self.model.queue.put(CORPUS_LOADED_EVENT)
            except Exception as e:
                print(e)
                self.model.queue.put(ERROR_LOADING_CORPUS_EVENT)