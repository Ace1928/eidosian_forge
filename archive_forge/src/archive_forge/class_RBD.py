class RBD(object):

    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        return False

    def create(self, *args, **kwargs):
        pass

    def remove(self, *args, **kwargs):
        pass

    def list(self, *args, **kwargs):
        raise NotImplementedError()

    def clone(self, *args, **kwargs):
        raise NotImplementedError()