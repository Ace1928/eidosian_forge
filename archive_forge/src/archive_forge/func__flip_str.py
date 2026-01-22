from parlai.core.message import Message
from parlai.core.teachers import FixedDialogTeacher
from parlai.tasks.light_dialog.agents import DefaultTeacher as OrigLightTeacher
from parlai.tasks.light_genderation_bias.build import build
from collections import deque
from copy import deepcopy
import csv
import json
import os
import random
def _flip_str(self, txt_lines):
    new_lines = []
    lines = txt_lines.split('\n')
    for text in lines:
        f_text = format_text(text)
        f_text_lst = f_text.split(' ')
        new_words = []
        for word in f_text_lst:
            if word in self.swap_dct:
                if word == 'her':
                    random_choice = random.choice([0, 1])
                    if random_choice:
                        new_word = 'his'
                    else:
                        new_word = 'him'
                else:
                    new_word = self.swap_dct[word]['word']
            else:
                new_word = word
            new_words.append(new_word)
        new_f_text = ' '.join(new_words)
        uf_text = unformat_text(new_f_text)
        new_lines.append(uf_text)
    return '\n'.join(new_lines)