class CyclicGraph(Exception):
    """
    Exception for cyclic graphs.
    """

    def __init__(self):
        Exception.__init__(self, 'Graph is cyclic')