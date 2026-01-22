import sys
import time
from heatclient._i18n import _
from heatclient.common import utils
import heatclient.exc as exc
from heatclient.v1 import events as events_mod
def _get_stack_name_from_links(event):
    links = {link.get('rel'): link.get('href') for link in getattr(event, 'links', [])}
    href = links.get('stack')
    if not href:
        return
    return href.split('/stacks/', 1)[-1].split('/')[0]