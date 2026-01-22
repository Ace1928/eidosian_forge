class HeaderView(object):
    """A Text View With a Header

    This view simply serializes the model and places the given
    header on top.

    :param header: the header (can be anything on which str() can be called)
    """

    def __init__(self, header):
        self.header = header

    def __call__(self, model):
        return f'{self.header}\n{model}'