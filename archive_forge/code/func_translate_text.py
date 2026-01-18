import re
import gc
from time import sleep
from lxml import html
from collections import OrderedDict
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
from bs4 import BeautifulSoup
from nltk import sent_tokenize
from .browser import get_chrome
def translate_text(text, lf, lt):
    text = ' '.join(text.split())
    parts = Queue()
    if len(text) <= 4000:
        parts.put(text)
    else:
        sents = sent_tokenize(text)
        part = ''
        for sent in sents:
            part = sent + ' '
            if len(part) > 4000:
                parts.put(part.strip())
                part = ''
        else:
            parts.put(part.strip())
    result = OrderedDict()
    max_workers = 2
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for _ in range(max_workers):
            executor.submit(worker, parts, lf, lt, result)
    translated_text = ' '.join([x for x in result.values()])
    gc.collect()
    return translated_text