from __future__ import annotations
import threading
class LockedSingleton(object):
    """
    This is a singleton that is locked to a single thread
    """
    __instance = None
    __instance_lock = threading.RLock()

    def __new__(cls):
        """
        If the instance is not created, then create it
        """
        if cls.__instance is None:
            with cls.__instance_lock:
                cls.__instance = super(LockedSingleton, cls).__new__(cls)
        return cls.__instance