from typing import List, Optional
class UserDatabase:
    """
    L{UserDatabase} holds a traditional POSIX user data in memory and makes it
    available via the same API as L{pwd}.

    @ivar _users: A C{list} of L{_UserRecord} instances holding all user data
        added to this database.
    """
    _users: List[_UserRecord]
    _lastUID: int = 10101
    _lastGID: int = 20202

    def __init__(self) -> None:
        self._users = []

    def addUser(self, username: str, password: str='password', uid: Optional[int]=None, gid: Optional[int]=None, gecos: str='', home: str='', shell: str='/bin/sh') -> None:
        """
        Add a new user record to this database.

        @param username: The value for the C{pw_name} field of the user
            record to add.

        @param password: The value for the C{pw_passwd} field of the user
            record to add.

        @param uid: The value for the C{pw_uid} field of the user record to
            add.

        @param gid: The value for the C{pw_gid} field of the user record to
            add.

        @param gecos: The value for the C{pw_gecos} field of the user record
            to add.

        @param home: The value for the C{pw_dir} field of the user record to
            add.

        @param shell: The value for the C{pw_shell} field of the user record to
            add.
        """
        if uid is None:
            uid = self._lastUID
            self._lastUID += 1
        if gid is None:
            gid = self._lastGID
            self._lastGID += 1
        newUser = _UserRecord(username, password, uid, gid, gecos, home, shell)
        self._users.append(newUser)

    def getpwuid(self, uid: int) -> _UserRecord:
        """
        Return the user record corresponding to the given uid.
        """
        for entry in self._users:
            if entry.pw_uid == uid:
                return entry
        raise KeyError()

    def getpwnam(self, name: str) -> _UserRecord:
        """
        Return the user record corresponding to the given username.
        """
        if not isinstance(name, str):
            raise TypeError(f'getpwuam() argument must be str, not {type(name)}')
        for entry in self._users:
            if entry.pw_name == name:
                return entry
        raise KeyError()

    def getpwall(self) -> List[_UserRecord]:
        """
        Return a list of all user records.
        """
        return self._users