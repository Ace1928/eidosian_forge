class LayoutParameter(object):
    """
    Representation of a single HIT layout parameter
    """

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def get_as_params(self):
        params = {'Name': self.name, 'Value': self.value}
        return params