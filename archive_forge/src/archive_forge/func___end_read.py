from threading import Lock
def __end_read(self):
    lock = self.__lock
    lock.acquire()
    self.__visit_count = self.__visit_count - 1
    if self.__visit_count == 0:
        pending_removes = self.__pending_removes
        while pending_removes:
            s, p, o = pending_removes.pop()
            try:
                self.store.remove((s, p, o))
            except:
                print(s, p, o, 'Not in store to remove')
        pending_adds = self.__pending_adds
        while pending_adds:
            s, p, o = pending_adds.pop()
            self.store.add((s, p, o))
    lock.release()