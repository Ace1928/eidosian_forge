import copy
import datasets
import itertools
def to_dialog(thread):
    dialog = []
    for i, content in enumerate(thread):
        dialog.append({'role': 'user' if i % 2 == 0 else 'assistant', 'content': content})
    return {'dialog': dialog}