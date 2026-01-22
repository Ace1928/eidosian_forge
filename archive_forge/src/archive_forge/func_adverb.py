import random
from .english import adverbs, adjectives, names
def adverb(letters=6):
    while 1:
        w = random.choice(adverbs)
        if len(w) <= letters:
            return w