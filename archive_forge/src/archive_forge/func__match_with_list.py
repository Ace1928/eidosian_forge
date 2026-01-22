import uuid
from keystone.common import provider_api
import keystone.conf
from keystone import exception
def _match_with_list(self, this_batch, total_list, batch_size=None, list_start=None, list_end=None):
    if batch_size is None:
        batch_size = len(this_batch)
    if list_start is None:
        list_start = 0
    if list_end is None:
        list_end = len(total_list)
    for batch_item in range(0, batch_size):
        found = False
        for list_item in range(list_start, list_end):
            if this_batch[batch_item]['id'] == total_list[list_item]['id']:
                found = True
        self.assertTrue(found)