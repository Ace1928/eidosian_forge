from typing import List, Optional
class ShadowDatabase:
    """
    L{ShadowDatabase} holds a shadow user database in memory and makes it
    available via the same API as C{spwd}.

    @ivar _users: A C{list} of L{_ShadowRecord} instances holding all user data
        added to this database.

    @since: 12.0
    """
    _users: List[_ShadowRecord]

    def __init__(self) -> None:
        self._users = []

    def addUser(self, username: str, password: str, lastChange: int, min: int, max: int, warn: int, inact: int, expire: int, flag: int) -> None:
        """
        Add a new user record to this database.

        @param username: The value for the C{sp_nam} field of the user record to
            add.

        @param password: The value for the C{sp_pwd} field of the user record to
            add.

        @param lastChange: The value for the C{sp_lstchg} field of the user
            record to add.

        @param min: The value for the C{sp_min} field of the user record to add.

        @param max: The value for the C{sp_max} field of the user record to add.

        @param warn: The value for the C{sp_warn} field of the user record to
            add.

        @param inact: The value for the C{sp_inact} field of the user record to
            add.

        @param expire: The value for the C{sp_expire} field of the user record
            to add.

        @param flag: The value for the C{sp_flag} field of the user record to
            add.
        """
        self._users.append(_ShadowRecord(username, password, lastChange, min, max, warn, inact, expire, flag))

    def getspnam(self, username: str) -> _ShadowRecord:
        """
        Return the shadow user record corresponding to the given username.
        """
        if not isinstance(username, str):
            raise TypeError(f'getspnam() argument must be str, not {type(username)}')
        for entry in self._users:
            if entry.sp_nam == username:
                return entry
        raise KeyError(username)

    def getspall(self):
        """
        Return a list of all shadow user records.
        """
        return self._users