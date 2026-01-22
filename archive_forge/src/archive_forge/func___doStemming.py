import re
from nltk.stem.api import StemmerI
def __doStemming(self, word, intact_word):
    """Perform the actual word stemming"""
    valid_rule = re.compile('^([a-z]+)(\\*?)(\\d)([a-z]*)([>\\.]?)$')
    proceed = True
    while proceed:
        last_letter_position = self.__getLastLetter(word)
        if last_letter_position < 0 or word[last_letter_position] not in self.rule_dictionary:
            proceed = False
        else:
            rule_was_applied = False
            for rule in self.rule_dictionary[word[last_letter_position]]:
                rule_match = valid_rule.match(rule)
                if rule_match:
                    ending_string, intact_flag, remove_total, append_string, cont_flag = rule_match.groups()
                    remove_total = int(remove_total)
                    if word.endswith(ending_string[::-1]):
                        if intact_flag:
                            if word == intact_word and self.__isAcceptable(word, remove_total):
                                word = self.__applyRule(word, remove_total, append_string)
                                rule_was_applied = True
                                if cont_flag == '.':
                                    proceed = False
                                break
                        elif self.__isAcceptable(word, remove_total):
                            word = self.__applyRule(word, remove_total, append_string)
                            rule_was_applied = True
                            if cont_flag == '.':
                                proceed = False
                            break
            if rule_was_applied == False:
                proceed = False
    return word