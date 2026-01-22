import os
import sys
from os.path import pardir, realpath
def _print_dict(title, data):
    for index, (key, value) in enumerate(sorted(data.items())):
        if index == 0:
            print(f'{title}: ')
        print(f'\t{key} = "{value}"')