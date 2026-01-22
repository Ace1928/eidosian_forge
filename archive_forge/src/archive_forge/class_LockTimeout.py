import datetime
class LockTimeout(Exception):
    """An exception when a lock could not be acquired before a timeout period"""