import gc
import sys
import threading
from oslo_reports.models import threading as tm
from oslo_reports.models import with_default_views as mwdv
from oslo_reports.views.text import generic as text_views
def _find_objects(t):
    """Find Objects in the GC State

    This horribly hackish method locates objects of a
    given class in the current python instance's garbage
    collection state.  In case you couldn't tell, this is
    horribly hackish, but is necessary for locating all
    green threads, since they don't keep track of themselves
    like normal threads do in python.

    :param class t: the class of object to locate
    :rtype: list
    :returns: a list of objects of the given type
    """
    return [o for o in gc.get_objects() if isinstance(o, t)]