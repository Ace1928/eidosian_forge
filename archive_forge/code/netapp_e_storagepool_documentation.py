from __future__ import absolute_import, division, print_function
import functools
from itertools import groupby
from time import sleep
from pprint import pformat
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.netapp import NetAppESeriesModule
from ansible.module_utils._text import to_native
Perform the request for expanding existing volume groups or disk pools.

            Note: the list of candidate structures do not necessarily produce candidates that meet all criteria.
            