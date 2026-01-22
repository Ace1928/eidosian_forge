class OddInstance:

    def __init__(self, cls):
        self.__dict__['__class__'] = cls

    def __getattribute__(self, name):
        dict = object.__getattribute__(self, '__dict__')
        if name == '__dict__':
            return dict
        v = dict.get(name, self)
        if v is not self:
            return v
        return getattr(dict['__class__'], name)

    def __setattr__(self, name, v):
        self.__dict__[name] = v

    def __delattr__(self, name):
        raise NotImplementedError()

    def __repr__(self):
        return '<odd {} instance at {}>'.format(self.__class__.__name__, hex(id(self)))