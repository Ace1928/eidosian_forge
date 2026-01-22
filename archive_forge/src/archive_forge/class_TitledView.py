class TitledView(HeaderView):
    """A Text View With a Title

    This view simply serializes the model, and places
    a preformatted header containing the given title
    text on top.  The title text can be up to 64 characters
    long.

    :param str title:  the title of the view
    """
    FORMAT_STR = '=' * 72 + '\n===={0: ^64}====\n' + '=' * 72

    def __init__(self, title):
        super(TitledView, self).__init__(self.FORMAT_STR.format(title))