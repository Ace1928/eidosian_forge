from os import environ
from kivy.logger import Logger
from kivy.clock import Clock
@staticmethod
def _purge_oldest(category, maxpurge=1):
    Logger.trace('Cache: Remove oldest in %s' % category)
    import heapq
    time = Clock.get_time()
    heap_list = []
    for key in Cache._objects[category]:
        obj = Cache._objects[category][key]
        if obj['lastaccess'] == obj['timestamp'] == time:
            continue
        heapq.heappush(heap_list, (obj['lastaccess'], key))
        Logger.trace('Cache: <<< %f' % obj['lastaccess'])
    n = 0
    while n <= maxpurge:
        try:
            n += 1
            lastaccess, key = heapq.heappop(heap_list)
            Logger.trace('Cache: %d => %s %f %f' % (n, key, lastaccess, Clock.get_time()))
        except Exception:
            return
        Cache.remove(category, key)