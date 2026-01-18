import queue as q
import re
import threading
from tkinter import (
from tkinter.font import Font
from nltk.corpus import (
from nltk.draw.util import ShowText
from nltk.util import in_idle
def words_and_labels(self, sentence, pos1, pos2):
    search_exp = sentence[pos1:pos2]
    words, labels = ([], [])
    labeled_words = search_exp.split(' ')
    index = 0
    for each in labeled_words:
        if each == '':
            index += 1
        else:
            word, label = each.split('/')
            words.append((self._char_before + index, self._char_before + index + len(word)))
            index += len(word) + 1
            labels.append((self._char_before + index, self._char_before + index + len(label)))
            index += len(label)
        index += 1
    return (words, labels)