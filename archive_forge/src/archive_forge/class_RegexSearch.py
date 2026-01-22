import datetime
import hashlib
import heapq
import math
import os
import random
import re
import sys
import threading
import zlib
from peewee import format_date_time
@table_function(STRING)
class RegexSearch(TableFunction):
    params = ['regex', 'search_string']
    columns = ['match']
    name = 'regex_search'

    def initialize(self, regex=None, search_string=None):
        self._iter = re.finditer(regex, search_string)

    def iterate(self, idx):
        return (next(self._iter).group(0),)