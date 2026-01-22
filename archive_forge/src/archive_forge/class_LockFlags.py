import enum
import os
class LockFlags(enum.IntFlag):
    EXCLUSIVE = LOCK_EX
    SHARED = LOCK_SH
    NON_BLOCKING = LOCK_NB
    UNBLOCK = LOCK_UN