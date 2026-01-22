import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
class ConcordanceSearchModel:

    def __init__(self, queue):
        self.queue = queue
        self.CORPORA = _CORPORA
        self.DEFAULT_CORPUS = _DEFAULT
        self.selected_corpus = None
        self.reset_query()
        self.reset_results()
        self.result_count = None
        self.last_sent_searched = 0

    def non_default_corpora(self):
        copy = []
        copy.extend(list(self.CORPORA.keys()))
        copy.remove(self.DEFAULT_CORPUS)
        copy.sort()
        return copy

    def load_corpus(self, name):
        self.selected_corpus = name
        self.tagged_sents = []
        runner_thread = self.LoadCorpus(name, self)
        runner_thread.start()

    def search(self, query, page):
        self.query = query
        self.last_requested_page = page
        self.SearchCorpus(self, page, self.result_count).start()

    def next(self, page):
        self.last_requested_page = page
        if len(self.results) < page:
            self.search(self.query, page)
        else:
            self.queue.put(SEARCH_TERMINATED_EVENT)

    def prev(self, page):
        self.last_requested_page = page
        self.queue.put(SEARCH_TERMINATED_EVENT)

    def reset_results(self):
        self.last_sent_searched = 0
        self.results = []
        self.last_page = None

    def reset_query(self):
        self.query = None

    def set_results(self, page, resultset):
        self.results.insert(page - 1, resultset)

    def get_results(self):
        return self.results[self.last_requested_page - 1]

    def has_more_pages(self, page):
        if self.results == [] or self.results[0] == []:
            return False
        if self.last_page is None:
            return True
        return page < self.last_page

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

    class SearchCorpus(threading.Thread):

        def __init__(self, model, page, count):
            self.model, self.count, self.page = (model, count, page)
            threading.Thread.__init__(self)

        def run(self):
            q = self.processed_query()
            sent_pos, i, sent_count = ([], 0, 0)
            for sent in self.model.tagged_sents[self.model.last_sent_searched:]:
                try:
                    m = re.search(q, sent)
                except re.error:
                    self.model.reset_results()
                    self.model.queue.put(SEARCH_ERROR_EVENT)
                    return
                if m:
                    sent_pos.append((sent, m.start(), m.end()))
                    i += 1
                    if i > self.count:
                        self.model.last_sent_searched += sent_count - 1
                        break
                sent_count += 1
            if self.count >= len(sent_pos):
                self.model.last_sent_searched += sent_count - 1
                self.model.last_page = self.page
                self.model.set_results(self.page, sent_pos)
            else:
                self.model.set_results(self.page, sent_pos[:-1])
            self.model.queue.put(SEARCH_TERMINATED_EVENT)

        def processed_query(self):
            new = []
            for term in self.model.query.split():
                term = re.sub('\\.', '[^/ ]', term)
                if re.match('[A-Z]+$', term):
                    new.append(BOUNDARY + WORD_OR_TAG + '/' + term + BOUNDARY)
                elif '/' in term:
                    new.append(BOUNDARY + term + BOUNDARY)
                else:
                    new.append(BOUNDARY + term + '/' + WORD_OR_TAG + BOUNDARY)
            return ' '.join(new)