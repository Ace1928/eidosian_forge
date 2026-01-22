import re
from textwrap import wrap
from nltk.data import load
def _print_entries(tags, tagdict):
    for tag in tags:
        entry = tagdict[tag]
        defn = [tag + ': ' + entry[0]]
        examples = wrap(entry[1], width=75, initial_indent='    ', subsequent_indent='    ')
        print('\n'.join(defn + examples))