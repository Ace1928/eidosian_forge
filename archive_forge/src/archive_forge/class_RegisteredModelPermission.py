class RegisteredModelPermission:

    def __init__(self, name, user_id, permission):
        self._name = name
        self._user_id = user_id
        self._permission = permission

    @property
    def name(self):
        return self._name

    @property
    def user_id(self):
        return self._user_id

    @property
    def permission(self):
        return self._permission

    @permission.setter
    def permission(self, permission):
        self._permission = permission

    def to_json(self):
        return {'name': self.name, 'user_id': self.user_id, 'permission': self.permission}

    @classmethod
    def from_json(cls, dictionary):
        return cls(name=dictionary['name'], user_id=dictionary['user_id'], permission=dictionary['permission'])