from os.path import exists
import sys
from collections import defaultdict
import json
def write_special_section(fh, items, header):
    items = sorted(items, key=lambda x: x[0])
    if items:
        fh.write(f'{header}\n{'-' * len(header)}\n\n')
        for n, title in items:
            fh.write(f'- [:repo:`{n}`]: {title}\n')
        fh.write('\n')