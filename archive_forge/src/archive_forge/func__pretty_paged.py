import copy
import json
import optparse
import os
import pickle
import sys
from urllib import parse
from troveclient.compat import client
from troveclient.compat import exceptions
def _pretty_paged(self, func, *args, **kwargs):
    try:
        limit = self.limit
        if limit:
            limit = int(limit, 10)
        result = func(*args, limit=limit, marker=self.marker, **kwargs)
        if self.verbose:
            return
        if result and len(result) > 0:
            for item in result:
                print(self._dumps(item._info))
            if result.links:
                print('Links:')
                for link in result.links:
                    print(self._dumps(link))
        else:
            print('OK')
    except Exception:
        if self.debug:
            raise
        print(sys.exc_info()[1])