class Alternative(Symbol):
    """Unions"""

    def __init__(self, symbols, labels, default=NO_DEFAULT):
        super().__init__(symbols, default)
        self.labels = labels

    def get_symbol(self, index):
        return self.production[index]

    def get_label(self, index):
        return self.labels[index]