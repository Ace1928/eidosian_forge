from __future__ import print_function
import os,stat,time
import errno
import sys
def waitget(self, key, maxwaittime=60):
    """ Wait (poll) for a key to get a value

        Will wait for `maxwaittime` seconds before raising a KeyError.
        The call exits normally if the `key` field in db gets a value
        within the timeout period.

        Use this for synchronizing different processes or for ensuring
        that an unfortunately timed "db['key'] = newvalue" operation
        in another process (which causes all 'get' operation to cause a
        KeyError for the duration of pickling) won't screw up your program
        logic.
        """
    wtimes = [0.2] * 3 + [0.5] * 2 + [1]
    tries = 0
    waited = 0
    while 1:
        try:
            val = self[key]
            return val
        except KeyError:
            pass
        if waited > maxwaittime:
            raise KeyError(key)
        time.sleep(wtimes[tries])
        waited += wtimes[tries]
        if tries < len(wtimes) - 1:
            tries += 1