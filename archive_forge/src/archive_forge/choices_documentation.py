from collections.abc import Callable, Iterable, Iterator, Mapping
from itertools import islice, tee, zip_longest
from django.utils.functional import Promise
Normalize choices values consistently for fields and widgets.