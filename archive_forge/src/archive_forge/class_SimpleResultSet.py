class SimpleResultSet(list):
    """
    ResultSet facade built from a simple list, rather than via XML parsing.
    """

    def __init__(self, input_list):
        for x in input_list:
            self.append(x)
        self.is_truncated = False