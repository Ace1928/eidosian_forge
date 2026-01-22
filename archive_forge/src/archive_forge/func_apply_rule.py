from nltk.data import load
from nltk.stem.api import StemmerI
def apply_rule(self, word, rule_index):
    rules = self._model[rule_index]
    for rule in rules:
        suffix_length = len(rule[0])
        if word[-suffix_length:] == rule[0]:
            if len(word) >= suffix_length + rule[1]:
                if word not in rule[3]:
                    word = word[:-suffix_length] + rule[2]
                    break
    return word