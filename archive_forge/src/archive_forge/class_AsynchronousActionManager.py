import enum
class AsynchronousActionManager(object):

    @staticmethod
    def _flatten(*args):
        ahs = set()
        if len(args) > 0:
            for item in args:
                if type(item) is ActionHandle:
                    ahs.add(item)
                elif type(item) in (list, tuple, dict, set):
                    for ah in item:
                        if type(ah) is not ActionHandle:
                            raise ActionManagerError('Bad argument type %s' % str(ah))
                        ahs.add(ah)
                else:
                    raise ActionManagerError('Bad argument type %s' % str(item))
        return ahs

    def __init__(self):
        """Constructor"""
        self.clear()

    def clear(self):
        """
        Clear manager state
        """
        self.event_handle = {}
        self.results = {}
        self.queued_action_counter = 0

    def execute(self, *args, **kwds):
        """
        Synchronously execute an action.
        """
        ah = self.queue(*args, **kwds)
        results = self.wait_for(ah)
        if results is None:
            raise ActionManagerError('Problem executing an event.  No results are available.')
        return results

    def queue(self, *args, **kwds):
        """
        Queue an action, returning an ActionHandle object.
        """
        ah = ActionHandle()
        self.event_handle[ah.id] = ah
        ah.status = ActionStatus.queued
        self.queued_action_counter += 1
        return self._perform_queue(ah, *args, **kwds)

    def wait_all(self, *args):
        """
        Wait for all actions to complete.  The arguments to this method
        are expected to be ActionHandle objects or iterators that return
        ActionHandle objects.  If no arguments are provided, then this
        method will terminate after all queued actions are
        """
        ahs = set()
        if len(args) == 0:
            ahs.update((ah for ah in self.event_handle.values() if ah.status == ActionStatus.queued))
        else:
            ahs = self._flatten(*args)
        while len(ahs) > 0:
            ah = self.wait_any()
            ahs.discard(ah)

    def wait_any(self, *args):
        """
        Wait for any action (or any of the specified actions) to
        complete, and return the corresponding ActionHandle.
        """
        ah = None
        if len(args):
            ahs = self._flatten(*args)
            ah = None
            while ah not in ahs:
                ah = self._perform_wait_any()
        else:
            while ah is None:
                ah = self._perform_wait_any()
        if ah == FailedActionHandle:
            return ah
        self.queued_action_counter -= 1
        self.event_handle[ah.id].update(ah)
        return self.event_handle[ah.id]

    def wait_for(self, ah):
        """
        Wait for the specified action to complete.
        """
        tmp = self.wait_any()
        while tmp != ah:
            tmp = self.wait_any()
            if tmp == FailedActionHandle:
                raise ActionManagerError('Action %s failed: %s' % (ah, tmp.explanation))
        return self.get_results(ah)

    def num_queued(self):
        """
        Return the number of queued actions.
        """
        return self.queued_action_counter

    def get_status(self, ah):
        """
        Return the status of the ActionHandle.
        """
        return ah.status

    def get_results(self, ah):
        """
        Return solver results.  If solver results are not available,
        return None.
        """
        if ah.id in self.results:
            result = self.results[ah.id]
            del self.results[ah.id]
            return result
        return None

    def _perform_queue(self, ah, *args, **kwds):
        """
        Perform the queue operation.  This method returns the
        ActionHandle, and the ActionHandle status indicates whether
        the queue was successful.
        """
        raise ActionManagerError('The _perform_queue method is not defined')

    def _perform_wait_any(self):
        """
        Perform the wait_any operation.  This method returns an
        ActionHandle with the results of waiting.  If None is returned
        then the ActionManager assumes that it can call this method
        again.  Note that an ActionHandle can be returned with a dummy
        value, to indicate an error.
        """
        raise ActionManagerError('The _perform_wait_any method is not defined')